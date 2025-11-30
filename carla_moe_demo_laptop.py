#!/usr/bin/env python3
"""
Carla DEMO – laptop / CPU – uses EXACT training CFG (224×224, 64-ch LiDAR, 64×64 BEV)
-------------------------------------------------------------------------------------
python carla_moe_demo_laptop.py
python carla_moe_demo_laptop.py --show-cam   # + opencv window
Ctrl-C to quit.
"""
import cv2
import carla
import torch
import torchvision.transforms as T
import numpy as np
import pathlib, argparse, time
import pyttsx3




engine = pyttsx3.init()
engine.setProperty('rate', 180)

def speak(text):
    engine.say(text)
    engine.runAndWait()


STOP_ENTER_TIME = None   # epoch when we arrived at line
MAX_STOP_SEC    = 3.0    # demo timeout
ok_cnt = total_cnt = 0


def leading_vehicle(lidar_np, thresh=20.0):
    if lidar_np is None: return False
    # points in front quadrant
    front = lidar_np[(lidar_np[:, 0] > 0) & (np.abs(lidar_np[:, 1]) < 2.0)]
    return np.min(np.hypot(front[:, 0], front[:, 1])) < thresh

# ---------- road-code ----------
def get_speed_limit(vehicle):
    """km/h → m/s"""
    return vehicle.get_speed_limit() / 3.6

def traffic_light_state(vehicle):
    """return True if we must stop now"""
    light = vehicle.get_traffic_light()
    if light is None:
        return False
    return light.state == carla.TrafficLightState.Red

def stop_line_ahead(vehicle, wp, dist=3.5):
    """True if a stop-line is < dist m ahead"""
    for sign in wp.get_landmarks_of_type(dist, 'STOP'):
        loc = sign.transform.location
        if (loc - vehicle.get_location()).dot(wp.transform.get_forward_vector()) > 0:
            return True
    return False

def stop_sign_ahead(vehicle, world, dist=15.0):
    """
    True if a *traffic-sign* STOP actor is within <dist> m in front of us.
    """
    hero_tf = vehicle.get_transform()
    fwd     = hero_tf.get_forward_vector()
    hero_loc= hero_tf.location

    for sign in world.get_actors().filter('static.prop.trafficstop'):
        sign_loc = sign.get_location()
        to_sign  = sign_loc - hero_loc
        # same direction and close enough
        if to_sign.dot(fwd) > 0 and to_sign.length() < dist:
            return True
    return False


def _is_urban(hero):
    """
    Heuristic: if the speed-limit sign on this road is ≤ 50 km/h we call it urban.
    """
    return hero.get_speed_limit() <= 50.0  

def _agents_around(hero, world, radius=25.0, min_count=6):
    """True if ≥ min_count non-hero vehicles inside radius."""
    hero_loc = hero.get_location()
    def inside(a):
        return a.id != hero.id and a.get_location().distance(hero_loc) < radius
    return sum(1 for a in world.get_actors().filter('vehicle.*') if inside(a)) >= min_count

def adaptive_speed_limit(hero, world, wp, raw_red, lidar_data):
    """
    Returns the effective speed-limit (m/s) that the controller must respect.
    Hierarchy (smallest wins):
      1. Junction / stop-line / red light      15 km/h  (4.2 m/s)
      2. Crowded road                          20 km/h  (5.6 m/s)
      3. Urban / residential                   25 km/h  (6.9 m/s)
      4. Carla speed-limit sign                (already in m/s)
    """
    carla_limit = get_speed_limit(hero)          # m/s, from road sign
    urban_limit = 25.0 / 3.6                     # 25 km/h
    crowd_limit = 20.0 / 3.6                     # 20 km/h
    junction_limit = 15.0 / 3.6                  # 15 km/h

    # 1. junction / stop / red
    if raw_red or stop_line_ahead(hero, wp, 5.0):
        return junction_limit

    # 2. crowded
    if _agents_around(hero, world):
        return crowd_limit

    # 3. urban
    if _is_urban(hero):
        return min(carla_limit, urban_limit)

    # 4. default – respect sign
    return carla_limit
# ----------------------------------------------------------

# ---------- inside the main loop ----------

def draw_minimap(vehicle, wp, size=100, scale=0.2):
    map_img = np.zeros((size, size, 3), dtype=np.uint8)
    hero_loc = vehicle.get_location()
    for i in range(1, 6):
        nxt = wp.next(i * 2.0)[0]
        px = int(size//2 + (nxt.transform.location.x - hero_loc.x) * scale)
        py = int(size//2 - (nxt.transform.location.y - hero_loc.y) * scale)
        cv2.circle(map_img, (px, py), 2, (255, 100, 0), -1)
    cv2.circle(map_img, (size//2, size//2), 3, (0, 255, 0), -1)
    return map_img


fourcc = cv2.VideoWriter_fourcc(*'mp4v')   # codec
out = None   
# ---------- YOUR TRAINING CFG ----------
CFG = {
    'img_size': 224, 'lidar_bev': 64, 'state_dim': 10,
    'n_experts': 5, 'top_k': 2,
    'latent': 256,
    'lidar_channels': 64,   # ← your real value
    'out_dim': 60,          # 30×2
    'device': "cpu",        # laptop
}
WEIGHTS_PATH = pathlib.Path("best.pt")  # << change if different
# ----------------------------------------

def obstacle_ahead(lidar_np, thresh=15.0):
    if lidar_np is None or lidar_np.size == 0:
        return False
    # use raw (N,3) and check horizontal distance
    horiz_dist = np.hypot(lidar_np[:, 0], lidar_np[:, 1])
    return np.min(horiz_dist) < thresh

# ---------- MODEL (exact clone of training) ----------
class Expert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision.models as models
        self.img_enc = models.resnet18(weights=None)
        self.img_enc.fc = torch.nn.Identity()  # (B,512)

        self.lidar_enc = torch.nn.Sequential(
            torch.nn.Conv2d(CFG['lidar_channels'], 32, 3, 2, 1), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, 2, 1), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, 2, 1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten()  # (B,128)
        )
        self.state_enc = torch.nn.Sequential(
            torch.nn.Linear(CFG['state_dim'], 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 128)
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(512 + 128 + 128, 512), torch.nn.ReLU(),
            torch.nn.Linear(512, CFG['latent'])
        )

    def forward(self, img, lidar, state):
        f1 = self.img_enc(img)
        f2 = self.lidar_enc(lidar)
        f3 = self.state_enc(state)
        return self.head(torch.cat([f1, f2, f3], dim=1))


class Gating(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(CFG['latent'], 256), torch.nn.ReLU(),
            torch.nn.Linear(256, CFG['n_experts'])
        )
        self.gate_out = None

    def forward(self, x):
        logits = self.net(x)
        top_logits, top_idx = torch.topk(logits, CFG['top_k'], dim=1)
        top_gate = torch.softmax(top_logits, dim=1)
        gates = torch.zeros_like(logits).scatter_(1, top_idx, top_gate)
        self.gate_out = gates
        return gates, top_idx


class MoEDriving(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = torch.nn.ModuleList([Expert() for _ in range(CFG['n_experts'])])
        self.gate = Gating()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(CFG['latent'], 128), torch.nn.ReLU(),
            torch.nn.Linear(128, CFG['out_dim'])
        )

    def forward(self, img, lidar, state):
        exp_outs = torch.stack([exp(img, lidar, state) for exp in self.experts], dim=1)
        gate_in = exp_outs.mean(dim=1)
        gates, idx = self.gate(gate_in)
        out = (exp_outs * gates.unsqueeze(-1)).sum(dim=1)
        return self.decoder(out)
# ----------------------------------------------------


# ---------- PRE-PROCESS (CPU) ----------
img_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((CFG['img_size'], CFG['img_size'])),
    T.ToTensor(),  # (3,224,224)
])


def lidar_to_bev(lidar_np, size=64, ch=64, x_range=(-40, 40), y_range=(-40, 40), z_range=(-3, 5)):
    """
    LiDAR (N, 3) → (ch, size, size)  occupancy grid
    """
    if lidar_np is None or lidar_np.size == 0:
        return torch.zeros(ch, size, size)

    # filter points inside desired cube
    mask = (
        (lidar_np[:, 0] > x_range[0]) & (lidar_np[:, 0] < x_range[1]) &
        (lidar_np[:, 1] > y_range[0]) & (lidar_np[:, 1] < y_range[1]) &
        (lidar_np[:, 2] > z_range[0]) & (lidar_np[:, 2] < z_range[1])
    )
    pts = lidar_np[mask]                      # (M, 3)

    # discrete voxel indices
    dx = (pts[:, 0] - x_range[0]) / (x_range[1] - x_range[0]) * size
    dy = (pts[:, 1] - y_range[0]) / (y_range[1] - y_range[0]) * size
    dz = (pts[:, 2] - z_range[0]) / (z_range[1] - z_range[0]) * ch

    dx = np.clip(dx.astype(np.int32), 0, size - 1)
    dy = np.clip(dy.astype(np.int32), 0, size - 1)
    dz = np.clip(dz.astype(np.int32), 0, ch - 1)

    # occupancy grid
    bev = np.zeros((ch, size, size), dtype=np.float32)
    for ix, iy, iz in zip(dx, dy, dz):
        bev[iz, iy, ix] = 1.0   # simple binary occupancy

    return torch.from_numpy(bev)
# ---------------------------------------


# ---------- CARLA LOW-SPEC ----------
def carla_init(host="localhost", port=2000):
    client = carla.Client(host, port)
    client.set_timeout(60.0)
    world = client.load_world("Town05")

    weather_list = [
    carla.WeatherParameters.ClearNoon,
    carla.WeatherParameters.WetNoon,
    carla.WeatherParameters.HardRainNoon,
    carla.WeatherParameters.ClearNight
    ]
    idx = 0
    def next_weather():
        global idx
        idx = (idx + 1) % len(weather_list)
        world.set_weather(weather_list[idx])
        print("Weather →", weather_list[idx].name)

    if cv2.waitKey(1) & 0xFF == ord('w'):
        next_weather()


    settings = world.get_settings()
    settings.no_rendering_mode = False
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.10  # 10 Hz
    world.apply_settings(settings)
    # unload heavy layers
    world.unload_map_layer(carla.MapLayer.ParkedVehicles)
    world.unload_map_layer(carla.MapLayer.Props)
    world.set_weather(carla.WeatherParameters.ClearNoon)
    print("✓ Carla world ready (low-spec, 10 Hz)")
    return world


def spawn_hero_and_sensors(world):
    bp_lib = world.get_blueprint_library()
    hero_bp = bp_lib.filter('vehicle.tesla.model3')[0]
    hero_bp.set_attribute('role_name', 'hero')
    spawn = np.random.choice(world.get_map().get_spawn_points())
    hero = world.spawn_actor(hero_bp, spawn)
        # ---------- lane-keeping helper ----------
    prev_wp = world.get_map().get_waypoint(hero.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
    lane_offset = 0.0

    # RGB camera (full 224×224)
    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(CFG['img_size']))
    cam_bp.set_attribute('image_size_y', str(CFG['img_size']))
    cam_bp.set_attribute('fov', '90')
    camera = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.2, z=1.5)), attach_to=hero)

    # LiDAR (64 ch → you trained with this)
    lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('range', '85')
    lidar_bp.set_attribute('points_per_second', '224000')  # ¼ of default
    lidar_bp.set_attribute('rotation_frequency', '10')
    lidar = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(x=1.2, z=1.5)), attach_to=hero)

    return hero, camera, lidar
# ------------------------------------


# ---------- CONTROL ----------
def state_vector(vehicle):
    v = vehicle.get_velocity()
    acc = vehicle.get_acceleration()
    yaw = vehicle.get_transform().rotation.yaw
    return np.array([np.hypot(v.x, v.y), v.x, v.y, acc.x, acc.y, yaw, 0, 0, 0, 0], dtype=np.float32)


def controls_from_pred(pred_tensor, lane_offset, speed_limit_mps, must_stop, lidar_data, spd):
    pred = pred_tensor[0].cpu().numpy()

    # 1. STEERING – do NOT centre the vector, just smooth it
    steer_vec    = pred[:30]                       # absolute angles (rad)
    net_steer    = float(np.median(steer_vec[:5])) # short horizon smooth

    # 2. LANE-DEPARTURE TRIM – speed-sensitive gain
    Kp = 1.2 * max(0.3, 1.0 - spd/10.0)          # 1.2→0.36 as spd 0→10 m/s
    final_steer  = np.clip(net_steer - Kp * lane_offset, -0.25, 0.25)

    # 3. THROTTLE / BRAKE – unchanged
    throttle_vec = np.clip(pred[30:], 0, 1)
    net_throttle = float(np.median(throttle_vec[:3]))

    if spd < 0.5:                                  # launch assist
        final_throttle = 0.45
        speak("Launching")
    else:
        # ACC-like lead-vehicle cap
        if lidar_data is not None and spd > 1.0:
            front = lidar_data[(lidar_data[:, 0] > 0) & (np.abs(lidar_data[:, 1]) < 2.0)]
            if front.size > 0 and np.min(np.hypot(front[:, 0], front[:, 1])) < 20.0:
                speed_limit_mps = min(speed_limit_mps, 5.0)
                speak("Following vehicle")
        max_throttle = min(net_throttle, speed_limit_mps / 8.0)
        final_throttle = np.clip(max_throttle, 0.1, 0.7)

    brake = float(final_throttle < 0.05)
    return carla.VehicleControl(throttle=final_throttle,steer=final_steer,brake=brake)

# ---------- DEMO LOOP ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--show-cam", action="store_true", help="open opencv window (costs CPU)")
    args = parser.parse_args()

    global ok_cnt
    global total_cnt
    # model
    model = MoEDriving().to(CFG['device'])
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(WEIGHTS_PATH)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()
    print("✓ MoE loaded (CPU)")

    # carla
    world = carla_init(args.host, args.port)
    hero, camera, lidar = spawn_hero_and_sensors(world)
    print("✓ Hero ready – driving with MoE")

        # ---------- video recorder ----------
    if args.show_cam:
        out = cv2.VideoWriter('carla_moe_run_demo.mp4', fourcc, 10.0,
                              (CFG['img_size'], CFG['img_size']))
        print("Recording → carla_moe_run_demo.mp4")

    image_data, lidar_data = None, None

    def image_cb(image):
        nonlocal image_data
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(CFG['img_size'], CFG['img_size'], 4)
        image_data = array[:, :, :3]

    def lidar_cb(scan):
        nonlocal lidar_data
        lidar_data = np.copy(np.frombuffer(scan.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3])
        print("LiDAR pts:", lidar_data.shape[0])
    camera.listen(image_cb)
    lidar.listen(lidar_cb)

    try:
        print("Driving... (Ctrl-C to stop)")
        while True:
            world.tick()
            # ---------- lane-center error ----------
            curr_wp = world.get_map().get_waypoint(hero.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
            spd = np.hypot(hero.get_velocity().x, hero.get_velocity().y)
            lane_offset = (hero.get_location() - curr_wp.transform.location).dot(
                  curr_wp.transform.get_right_vector())
            if curr_wp is not None:
                lane_center = curr_wp.transform.location
                vehicle_loc = hero.get_location()
                # signed lateral distance (right = +)
                #lane_offset = (vehicle_loc - lane_center).dot(curr_wp.transform.get_right_vector())
                prev_wp = curr_wp
                                # ---------- road-code ----------
                speed_limit   = get_speed_limit(hero)          # m/s
                                # ---------- smart stop ----------
                raw_red = traffic_light_state(hero) or stop_line_ahead(hero, curr_wp, 4.0) or stop_sign_ahead(hero, world)
                        # ---------- road-code ----------
                speed_limit = adaptive_speed_limit(hero, world, curr_wp, raw_red, lidar_data)         # m/s
                #raw_red = traffic_light_state(hero) or stop_line_ahead(hero, curr_wp, 4.0)
                obstacle    = obstacle_ahead(lidar_data, thresh=12.0)   # 12 m
                now = time.time()

                # 1. emergency brake (highest priority)
                if obstacle:
                    must_stop_now = True
                    STOP_ENTER_TIME = None          # cancel red-light timer
                    speak("Obstacle ahead")         # optional
                # 2. red-light / stop-line with 3 s timeout
                elif raw_red:
                    if STOP_ENTER_TIME is None:     # just arrived
                        STOP_ENTER_TIME = now
                    must_stop_now = (now - STOP_ENTER_TIME) < MAX_STOP_SEC
                # 3. green / no line
                else:
                    STOP_ENTER_TIME = None
                    must_stop_now = False
            if image_data is None or lidar_data is None:
                continue

            # 2. (optional) one-time message when LiDAR wakes up
            if lidar_data is not None and lidar_data.size > 0:
                print("LiDAR alive – size:", lidar_data.shape)

            # preprocess
            img_t = img_transform(image_data).unsqueeze(0)
            lidar_t = lidar_to_bev(lidar_data).unsqueeze(0)
            state_t = torch.from_numpy(state_vector(hero)).unsqueeze(0).float()

            print("img  min/max:", img_t.min().item(), img_t.max().item())
            print("lidar min/max:", lidar_t.min().item(), lidar_t.max().item())
            print("state:", state_t)

            # inference
            with torch.no_grad():
                pred = model(img_t, lidar_t, state_t)

            # telemetry
            #spd = np.hypot(hero.get_velocity().x, hero.get_velocity().y)

            # control
            control = controls_from_pred(pred, lane_offset, speed_limit, must_stop_now, lidar_data,spd=spd)
            hero.apply_control(control)

            
            print(f"\rspd={spd:4.1f}  steer={control.steer:+5.2f}  throttle={control.throttle:4.2f}", end="", flush=True)

            # optional window
            if args.show_cam:
                       
                frame = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
                            # ---------- HUD ----------
                ok_cnt += 1 if abs(lane_offset) < 0.5 else 0
                total_cnt += 1
                acc = ok_cnt / max(total_cnt, 1)
                cv2.putText(frame, f"SPD {spd*3.6:3.0f} km/h", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"STR {control.steer:+.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"LANE ACC {acc:.1%}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                # traffic-light icon
                color = (0, 0, 255) if must_stop_now else (0, 255, 0)
                cv2.circle(frame, (200, 30), 15, color, -1)
                # mini-map (100×100) top-right
                mini = draw_minimap(hero, curr_wp)
                frame[0:100, -100:] = mini
                cv2.imshow("MoE cam", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # start recorder on first frame
                if out is None:
                    out = cv2.VideoWriter('carla_moe_run_demo.mp4', fourcc, 10.0,
                                          (frame.shape[1], frame.shape[0]))
                    print("🎥 Recording → carla_moe_run_demo.mp4")
                out.write(frame)

    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    finally:
        if args.show_cam:
            cv2.destroyAllWindows()
            if out is not None:
                out.release()
                print("✓ Video saved → carla_moe_run_demo.mp4")
        camera.stop()
        lidar.stop()
        hero.destroy()
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("Clean exit.")


if __name__ == "__main__":
    main()