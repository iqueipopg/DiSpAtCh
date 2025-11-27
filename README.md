# DiSpAtCh 🤖📦

**DiSpAtCh** (*Deep Intelligent Spatial Agent for Task CHaining*) is a Deep Reinforcement Learning system that trains an autonomous agent to navigate a warehouse environment, pick up objects from shelves, and deliver them to a designated area.

This project was developed as part of the **Reinforcement Learning** course at **Universidad Pontificia Comillas, ICAI**, within the **Engineering Mathematics (iMAT)** program.

> 🎯 *From random wandering to efficient delivery — DiSpAtCh learns warehouse logistics through trial and error.*

---

## 📜 Table of Contents
- [📌 Project Overview](#-project-overview)
- [🎮 Environments](#-environments)
- [🛠️ Installation](#️-installation)
- [🚀 How to Use](#-how-to-use)
- [📂 Project Structure](#-project-structure)
- [📊 Results](#-results)
- [🧠 Technologies Used](#-technologies-used)
- [🙌 Credits](#-credits)

---

## 📌 Project Overview

DiSpAtCh implements a **Double DQN (Deep Q-Network)** agent that learns to operate in a simulated warehouse environment. The agent must learn to:

1. **Navigate** around obstacles (shelves) without collisions
2. **Pick up** objects from fixed or random positions
3. **Deliver** objects to a designated green area

### 🧠 Key Features

- **Double DQN** with experience replay for stable learning
- **Dense feature representation** (32 features) optimized for neural networks
- **Reward shaping** to guide the agent toward objectives
- **Epsilon-greedy exploration** with linear decay
- **Best model checkpointing** during training

### 🏭 The Warehouse

```
    ┌─────────────────────────────┐
  10│      🟩 DELIVERY AREA 🟩     │
    │                             │
   8│                             │
    │                             │
   6│  ║     ║     ║              │
    │  ║     ║     ║              │
   4│  ║ 🔵  ║     ║  🔵          │
    │  ║     ║ 🔵  ║              │
   2│  ║     ║     ║              │
    │                             │
   0└─────────────────────────────┘
    0    2    4    6    8   10

    🟠 Agent    🔵 Objects    ║ Shelves    🟩 Delivery
```

---

## 🎮 Environments

The project includes three environments of increasing difficulty:

| Environment | Objects | Objective | Actions | Difficulty |
|-------------|---------|-----------|---------|------------|
| **Entorno 1** | Fixed positions | Pick only | 5 (↑↓←→ pick) | ⭐ |
| **Entorno 2** | Fixed positions | Pick + Delivery | 6 (↑↓←→ pick drop) | ⭐⭐ |
| **Entorno 3** | Random positions | Pick + Delivery | 6 (↑↓←→ pick drop) | ⭐⭐⭐ |

### Reward Structure

| Event | Reward |
|-------|--------|
| Each step | -0.02 |
| Collision (wall/shelf) | -15.0 |
| Successful pick | +100.0 |
| Successful delivery | +200.0 |
| Drop outside area | -20.0 |
| Timeout (1000 steps) | -10.0 |
| Moving toward objective | +0.05 / +0.5 |

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch
- Gymnasium

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/DiSpAtCh.git
   cd DiSpAtCh
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/macOS
   .\venv\Scripts\activate       # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Use

### Training

Train an agent on any of the three environments:

```bash
python dqn.py
```

Select the environment when prompted:
```
Select environment to train:
1. Entorno 1: Fixed objects, pick only
2. Entorno 2: Fixed objects, pick + delivery
3. Entorno 3: Random objects, pick + delivery
4. Train all (sequentially)
```

The best model will be automatically saved as `best_model_Entorno_X.pth`.

### Testing a Trained Model

Visualize how a trained agent performs:

```bash
# Test Environment 1
python test_model.py best_model_Entorno_1.pth --env 1

# Test Environment 2
python test_model.py best_model_Entorno_2.pth --env 2

# Slower visualization
python test_model.py best_model_Entorno_1.pth --delay 0.2

# More episodes
python test_model.py best_model_Entorno_1.pth --episodes 10
```

---

## 📂 Project Structure

```
DiSpAtCh/
│
├── dqn.py                      # Main training script (DQN agent)
├── test_model.py               # Script to visualize trained models
├── almacen_alu_v1.py           # Warehouse environment (Gymnasium)
├── representacion_wharehouse.py # Feature extraction (32 dense features)
├── tiles3.py                   # Tile coding utilities (optional)
│
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── models/                     # Saved models (generated during training)
│   ├── best_model_Entorno_1.pth
│   ├── best_model_Entorno_2.pth
│   └── best_model_Entorno_3.pth
│
└── images/                     # Images for documentation
    └── banner.png
```

---

## 📊 Results

### Environment 1: Pick Only

| Metric | Value |
|--------|-------|
| Success Rate | **100%** |
| Average Steps | ~10 |
| Training Episodes | ~1800 |

### Environment 2: Pick + Delivery

| Metric | Value |
|--------|-------|
| Success Rate | **TBD** |
| Average Steps | TBD |
| Training Episodes | TBD |

### Environment 3: Random Objects

| Metric | Value |
|--------|-------|
| Success Rate | **TBD** |
| Average Steps | TBD |
| Training Episodes | TBD |

---

## 🧠 Technologies Used

### Frameworks & Libraries

- **PyTorch** – Deep learning framework for DQN implementation
- **Gymnasium** – RL environment interface
- **NumPy** – Numerical computations
- **Matplotlib** – Training visualization and environment rendering
- **tqdm** – Progress bars during training

### RL Techniques

- **Double DQN** – Reduces overestimation bias in Q-learning
- **Experience Replay** – Breaks correlation between consecutive samples
- **Target Network** – Stabilizes training with periodic weight updates
- **Epsilon-Greedy** – Balances exploration vs exploitation
- **Reward Shaping** – Guides agent toward objectives

---

## 🙌 Credits

This project was developed as part of the **Reinforcement Learning** course at **Universidad Pontificia Comillas, ICAI**.

### Team Members

- **Beltrán Sánchez Careaga**
- **Jorge Kindelan Navarro**
- **Ignacio Queipo de Llano Pérez-Gascón**

### Acknowledgments

- Our professors for their guidance throughout the course
- The **PyTorch** and **Gymnasium** communities for excellent documentation
- **Rich Sutton** for the tile coding implementation (tiles3.py)

---

## 📄 License

This project is for educational purposes as part of the iMAT program at ICAI.
