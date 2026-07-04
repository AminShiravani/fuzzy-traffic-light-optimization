# Fuzzy Traffic Light Controller Optimization using PSO & ACO

This project implements a Mamdani Fuzzy Logic Controller (FLC) to optimize traffic light timings at a two-road intersection. To achieve optimal performance under varying traffic demands, the fuzzy membership functions and rule base parameters are optimized using two metaheuristic algorithms: **Particle Swarm Optimization (PSO)** and **Ant Colony Optimization (ACO)**.

---

## 📌 Project Overview
Urban traffic congestion is a dynamic, non-linear control problem. Standard fixed-time traffic controllers fail to adapt to real-time traffic fluctuations. This project models a discrete-time traffic simulation for a two-road intersection and evaluates three approaches:
1. **Baseline Mamdani Fuzzy Controller**: Uses pre-defined heuristic membership functions.
2. **PSO-Optimized Fuzzy Controller**: Tunes membership function parameters to minimize a joint traffic cost function.
3. **ACO-Optimized Fuzzy Controller**: Searches the optimal parameter configuration space to minimize the same cost function.

### Cost Function
The performance of each controller is evaluated using the following multi-objective cost function:
$$\text{Cost} = \alpha W + \beta Q + \gamma S$$
Where:
- $W$: Average waiting time of vehicles.
- $Q$: Average queue length at the intersection.
- $S$: Total number of stops.
- $\alpha, \beta, \gamma$: Weight factors adjusting the priority of each metric.

---

## 📂 Repository Structure
```
fuzzy-traffic-light-optimization/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 01_traffic_simulation.ipynb   # Time-discrete traffic intersection simulator
│   ├── 02_fuzzy_controller.ipynb    # Mamdani FLC implementation
│   ├── 03_pso_optimization.ipynb    # Tuning fuzzy system with PSO
│   ├── 04_aco_optimization.ipynb    # Tuning fuzzy system with ACO
│   └── 05_final_comparison.ipynb    # Benchmark results and visualizations
│
├── src/
│   ├── __init__.py
│   ├── simulation.py                # Intersection environment class
│   ├── fuzzy_controller.py          # Fuzzy inference system setup
│   ├── cost.py                      # Cost function calculator
│   ├── pso.py                       # PSO algorithm implementation
│   ├── aco.py                       # ACO algorithm implementation
│   └── plots.py                     # Visualization helpers
└── 
```

## 🛠️ Installation & Setup
Clone the Repository:
bash
   git clone https://github.com/AminShiravani/fuzzy-traffic-light-optimization.git
   cd fuzzy-traffic-light-optimization
   
Create and Activate Virtual Environment:
```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
Install Dependencies:
```bash
   pip install -r requirements.txt
   ```
## 🚀 Workflow & Execution
We follow a Notebook-first iterative development approach. You can run individual modules through Jupyter Notebooks or execute the clean scripts inside the src/ directory.

1. Run the Jupyter Notebooks
Start the Jupyter environment to explore the steps step-by-step:

bash
jupyter notebook
Open and run the notebooks inside the notebooks/ directory sequentially.

2. Running Simulations (Python Scripts)
Once the parameters are optimized, you can run the main simulation scripts:

```bash
python src/simulation.py
```

## 📊 Evaluation Metrics & Results
The final comparison analyzes:

Convergence Curves: Rate of cost reduction over generations/iterations for both PSO and ACO.
Queue Profiles: Traffic queue lengths on both roads over simulation steps.
Statistical Summary: Comparison table showing Average Wait Time, Max Queue Length, Stop Count, and Final Cost across all methods.
All plots and performance data are exported automatically to the results/ folder.

## 👥 Contributors
Amin Shiravani - Student in Software Analysis and Design / Computational Intelligence
Abolfazl Shahsavari - Student in Software Analysis and Design / Computational Intelligence
