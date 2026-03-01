<div align="center">

# 🚗 AURA — Autonomous Urban Reasoning Agent

### Mixture-of-Experts Architecture for Autonomous Vehicle Decision-Making

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![CARLA](https://img.shields.io/badge/CARLA-Simulator-1a1a2e?logo=car&logoColor=white)](https://carla.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-MoE%20Model-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 🏆 Built for the **Capgemini Engineering Innovathon**  
> Track: **Autonomous Vehicles** · Capgemini Engineering, Morocco

</div>

---

## 📖 Overview

**AURA** (Autonomous Urban Reasoning Agent) is an intelligent driving system that leverages a **Mixture-of-Experts (MoE)** neural architecture to handle complex, real-world autonomous driving scenarios. Rather than relying on a single monolithic model, AURA dynamically routes driving decisions — such as lane keeping, obstacle avoidance, and speed control — to specialized expert networks, each optimized for a distinct driving context.

The system is trained and validated using synthetic driving data generated from the **CARLA open-source autonomous driving simulator**, enabling safe, scalable, and reproducible experimentation.

---

## 🧠 What is Mixture-of-Experts (MoE)?

Traditional neural networks use all their parameters for every input. MoE changes this: it trains multiple **specialist sub-networks (experts)** and uses a **gating network** to decide which expert(s) are most relevant for each situation.

```
Input (Sensor Data)
        │
        ▼
  ┌─────────────┐
  │  Gating Net  │  ──► Selects relevant expert(s)
  └─────────────┘
        │
   ┌────┴─────┐
   ▼    ▼     ▼
 Exp1  Exp2  Exp3   ◄── Specialist networks
   └────┬─────┘
        ▼
  Weighted Output
  (Driving Decision)
```

This makes AURA more **efficient**, **modular**, and better at handling the **diverse scenarios** encountered in real urban driving.

---

## ✨ Features

- 🧩 **Mixture-of-Experts Architecture** — Dynamic expert routing for context-aware decisions
- 🏙️ **CARLA Simulator Integration** — Realistic urban driving data generation and live demo
- 📊 **Full Training Pipeline** — Data generation → model training → evaluation, all in notebooks
- 💻 **Laptop-Friendly Demo** — Optimized demo script for running on consumer hardware
- 🔬 **Research-Grade Notebooks** — Clean, well-structured Jupyter notebooks for reproducibility

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Simulator** | [CARLA](https://carla.org/) (Open-source Autonomous Driving Simulator) |
| **ML Framework** | PyTorch |
| **Architecture** | Mixture-of-Experts (MoE) Neural Network |
| **Notebooks** | Jupyter Notebook |
| **Language** | Python 3.10+ |

---

## 📁 Project Structure

```
Capgemini-AURA/
│
├── generate_carla_data.py       # Script to collect driving data from CARLA
├── AURA_MoE_training.ipynb      # Full MoE model training pipeline
├── AURA_MoE.ipynb               # Model architecture, evaluation & analysis
├── carla_moe_demo_laptop.py     # Lightweight real-time demo using CARLA
└── README.md
```

### File Roles at a Glance

**`generate_carla_data.py`** — Connects to a running CARLA instance and collects sensor data (camera, lidar, speed, steering) across various urban scenarios. The output feeds directly into the training pipeline.

**`AURA_MoE_training.ipynb`** — Defines the MoE architecture, trains the gating network and expert networks, and saves the final model weights. Start here for model development.

**`AURA_MoE.ipynb`** — Loads trained weights for inference, visualizes expert activation patterns, and evaluates performance across driving scenarios.

**`carla_moe_demo_laptop.py`** — A streamlined real-time demo that loads the trained AURA model and controls a vehicle inside CARLA, designed to run smoothly on a standard laptop.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [CARLA Simulator](https://carla.org/download/) (version 0.9.13+ recommended)
- A GPU is recommended for training (CPU works for the demo)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MohamedReda2003/Capgemini-AURA.git
   cd Capgemini-AURA
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   > If no `requirements.txt` is present, install manually:
   ```bash
   pip install torch numpy pandas matplotlib jupyter carla
   ```

3. **Launch CARLA Simulator**

   Start the CARLA server before running any script:
   ```bash
   # Linux
   ./CarlaUE4.sh

   # Windows
   CarlaUE4.exe
   ```

---

## 🎮 Usage

### Step 1 — Generate Training Data
```bash
python generate_carla_data.py
```
This will connect to the running CARLA instance, spawn a vehicle, and record sensor + control data into a dataset file.

### Step 2 — Train the MoE Model

Open and run the training notebook:
```bash
jupyter notebook AURA_MoE_training.ipynb
```
Follow the cells sequentially to preprocess data, define the MoE architecture, and train the model.

### Step 3 — Analyze & Evaluate

Open the analysis notebook to visualize expert routing and evaluate model performance:
```bash
jupyter notebook AURA_MoE.ipynb
```

### Step 4 — Run the Live Demo
```bash
python carla_moe_demo_laptop.py
```
This loads the trained model and drives an autonomous vehicle in real time inside the CARLA simulator.

---

## 🏆 Hackathon Context

> Built at the **Capgemini Engineering Innovathon**  
> **Track:** Autonomous Vehicles  
> **Organizer:** Capgemini Engineering, Morocco

AURA was developed as part of Capgemini Engineering's Innovathon challenge, which tasked participants with tackling real-world engineering problems in autonomous mobility using cutting-edge AI and simulation tools.

---

## 🤝 Contributing

Pull requests, improvements, and discussions are welcome!

1. Fork the repository
2. Create your branch: `git checkout -b feature/your-improvement`
3. Commit your changes: `git commit -m "Add your improvement"`
4. Push and open a Pull Request

---

<div align="center">

Made with ⚡ by [Mohamed Reda Zhar](https://github.com/MohamedReda2003) · Capgemini Engineering Innovathon · Morocco

</div>
