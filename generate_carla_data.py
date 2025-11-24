
import carla, os, pathlib, torch, random, numpy as np
import shutil
import json

LOCAL_TEMP = pathlib.Path("E:/Carla/temp/carla_data")
OUTPUT = pathlib.Path("H:/My Drive/AURA_MoE/data/carla_pt")
FRAMES_PER_EXPERT = 1500   # 3 M total
BATCH_SIZE        = 200       # frames per CARLA tick
SAVE_EVERY        = 100     # print every 1k frames
CHECKPOINT_FILE = LOCAL_TEMP / "progress.json"

def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {}

def save_checkpoint(expert, split, count):
    checkpoint = load_checkpoint()
    checkpoint[f"E{expert}_{split}"] = count
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)

def get_actual_count(expert, split):
    """Count actual frames in directory (handles both individual and batch files)"""
    directory = LOCAL_TEMP / f"E{expert}" / split
    
    if not directory.exists():
        return 0
    
    # Check for batch files first
    batch_files = list(directory.glob("*.pt"))
    if batch_files:
        total = 0
        for batch_file in batch_files:
            try:
                batch = torch.load(batch_file)
                total += len(batch)  # Count frames in batch
            except:
                print(f"Warning: Could not load {batch_file}, skipping")
        return total
    
    # Fallback: count individual .pt files
    individual_files = list(directory.glob("*.pt"))
    return len(individual_files)


def verify_and_load_checkpoint():
    """Load checkpoint and verify against actual files, return corrected counts"""
    checkpoint = load_checkpoint()
    verified = {}
    
    for e in range(5):
        for split in ["training", "validation"]:
            key = f"E{e}_{split}"
            checkpoint_count = checkpoint.get(key, 0)
            actual_count = get_actual_count(e, split)
            
            if checkpoint_count != actual_count:
                print(f"⚠️  {key}: Checkpoint says {checkpoint_count}, found {actual_count} frames. Using actual count.")
                verified[key] = actual_count
                # Update checkpoint file
                save_checkpoint(e, split, actual_count)
            else:
                verified[key] = actual_count
    
    return verified

for e in range(5):
    for split in ["training", "validation"]:
        (LOCAL_TEMP / f"E{e}" / split).mkdir(parents=True, exist_ok=True)

client = carla.Client("localhost", 2000)
client.set_timeout(60.0)
world  = client.load_world("Town05")
tm     = client.get_trafficmanager(8000)
weather_list = [
    carla.WeatherParameters.ClearNoon,
    carla.WeatherParameters.ClearNoon,
    carla.WeatherParameters.WetNoon,
    carla.WeatherParameters.HardRainNoon,
    carla.WeatherParameters.ClearNight
]

def get_weather(idx):
    return weather_list[idx % 5]

def tag_scenario(vehicle, world,i):
    """Extremely lenient - almost everything matches something"""
    if vehicle is None or not vehicle.is_alive:
        return -1
    
    try:
        v = vehicle.get_velocity()
        spd = np.hypot(v.x, v.y)
        
        # Distribute more evenly - use speed ranges
        if spd > 12:  # Fast
            return 0
        elif spd > 6:      return 3 if (i % 2 == 0) else 4    # Randomly assign to turns or default || medium speed
        elif spd > 2:      return 1 if (i % 2 == 0) else 4  # Randomly assign to cut-in or default  || medium slow
        else:  # Slow/stopped
            return 2
    except RuntimeError:
        return -1
    

def dummy_sensors(world, vehicle):
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    lidar = np.random.rand(3, 64, 64).astype(np.float32)
    v = vehicle.get_velocity()
    state = np.array([np.hypot(v.x, v.y), v.x, v.y,
                      vehicle.get_acceleration().x, vehicle.get_acceleration().y,
                      vehicle.get_transform().rotation.yaw, 0, 0, 0, 0], dtype=np.float32)
    future = np.random.rand(30, 2).astype(np.float32)  # dummy 3-s
    return image, lidar, state, future

def main():
    bp = world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
    spawn_points = world.get_map().get_spawn_points()
    checkpoint = verify_and_load_checkpoint()
    
    # Configure traffic manager for more varied behavior
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.set_synchronous_mode(True)
    
    FRAMES_PER_FILE = 100  # Save 100 frames per batch file
    
    for e in range(5):
        for split in ["training", "validation"]:
            key = f"E{e}_{split}"
            count = checkpoint.get(key, 0)
            
            if count >= FRAMES_PER_EXPERT:
                print(f"✓ {key} complete, skipping")
                continue
            
            print(f"Resuming {key} from {count} frames...")
            
            while count < FRAMES_PER_EXPERT:
                world.set_weather(get_weather(e))
                spawn = random.choice(spawn_points)
                try:
                    vehicle = world.spawn_actor(bp, spawn)
                except RuntimeError:
                    continue
                
                vehicle.set_autopilot(True, tm.get_port())
                tm.vehicle_percentage_speed_difference(vehicle, random.randint(-60, 40))
                tm.ignore_lights_percentage(vehicle, random.randint(10, 25))
                # Customize driving behavior per expert
                if e == 0:  # Highway
                    tm.vehicle_percentage_speed_difference(vehicle, -70)
                elif e == 1:  # Cut-ins
                    tm.vehicle_percentage_speed_difference(vehicle, -50)
                    tm.distance_to_leading_vehicle(vehicle, 0.5)
                    tm.set_random_device_seed(random.randint(0, 10000))
                elif e == 2:  # Urban
                    tm.vehicle_percentage_speed_difference(vehicle, 30)
                elif e == 3:  # Turns
                    tm.vehicle_percentage_speed_difference(vehicle, -20)
                    tm.auto_lane_change(vehicle, True)
                else:  # Expert 4
                    tm.vehicle_percentage_speed_difference(vehicle, 0)
                
                try:
                    batch_count = 0
                    non_match_count = 0
                    frame_buffer = []  # Buffer for batch saving
                    MAX_TICKS_PER_VEHICLE = 1500
                    
                    for i in range(MAX_TICKS_PER_VEHICLE):
                        try:
                            world.tick()
                        except RuntimeError as e:
                            print(f"  ⚠️  Simulation error at tick {i}: {e}")
                            break  # Exit loop, spawn new vehicle
                        e_tag = tag_scenario(vehicle, world,i)
                        
                        # Debug print every 100 ticks
                        if i % 100 == 0 and i > 0:
                            try:
                                v = vehicle.get_velocity()
                                spd = np.hypot(v.x, v.y)
                                print(f"  Tick {i}: e_tag={e_tag} (want {e}), speed={spd:.1f} m/s, v.y={abs(v.y):.2f}")
                            except RuntimeError:
                                print(f"  ⚠️  Vehicle destroyed during status check at tick {i}")
                                break  # Exit the for loop, go to finally block
                            print(f"  Tick {i}: e_tag={e_tag} (want {e}), speed={spd:.1f} m/s, v.y={abs(v.y):.2f}")
                        
                        if e_tag == e:
                            img, lidar, state, fut = dummy_sensors(world, vehicle)
                            
                            # Add to buffer instead of saving immediately
                            frame_buffer.append({
                                'image': img, 
                                'lidar': lidar, 
                                'state': state, 
                                'future': fut
                            })
                            batch_count += 1
                            
                            # Save when buffer reaches FRAMES_PER_FILE
                            if len(frame_buffer) >= FRAMES_PER_FILE:
                                batch_id = count // FRAMES_PER_FILE
                                filepath = LOCAL_TEMP / f"E{e}" / split / f"{batch_id:06d}.pt"
                                torch.save(frame_buffer, filepath)
                                count += len(frame_buffer)
                                frame_buffer = []  # Clear buffer
                                
                                # Save checkpoint
                                if count % 100 == 0:
                                    save_checkpoint(e, split, count)
                                
                                if count % SAVE_EVERY == 0:
                                    print(f"E{e} {split} {count//1000}k (match rate: {batch_count/(i+1):.1%})")
                            
                            if count >= FRAMES_PER_EXPERT:
                                break
                        else:
                            non_match_count += 1
                        
                        #if count >= FRAMES_PER_EXPERT or batch_count >= BATCH_SIZE:
                         #   break
                    
                    # Save any remaining frames in buffer
                    if frame_buffer and count < FRAMES_PER_EXPERT:
                        batch_id = count // FRAMES_PER_FILE
                        filepath = LOCAL_TEMP / f"E{e}" / split / f"{batch_id:06d}.pt"
                        torch.save(frame_buffer, filepath)
                        count += len(frame_buffer)
                        save_checkpoint(e, split, count)
                    
                    print(f"  Vehicle session: {batch_count} matches, {non_match_count} non-matches, {count} frames until now")
                    
                finally:
                    #vehicle.destroy()
                    #main()
                    if vehicle is not None:
                        try:
                            if vehicle.is_alive:
                                vehicle.destroy()
                        except:
                            pass  # Vehicle already destroyed or error - ignore
            
            save_checkpoint(e, split, count)
            print(f"✓ E{e} {split} finished – {count} frames")
            
            # After each split, MOVE entire directory
            print(f"  Moving E{e} {split} to Google Drive...")
            src = LOCAL_TEMP / f"E{e}" / split
            dst = OUTPUT / f"E{e}" / split
            
            if dst.exists():
                shutil.rmtree(dst)
            
            shutil.move(str(src), str(dst))
            print(f"  ✓ Move complete")
if __name__ == "__main__":
    main()
