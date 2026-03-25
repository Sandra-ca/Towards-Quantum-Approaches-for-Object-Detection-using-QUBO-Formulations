import matplotlib.pyplot as plt
import numpy as np
import os
  
# INITIALIZATION & X-AXIS SETUP  
x_labels = ["Low\n(2-20 box)", "Medium\n(21-60 box)", "High\n(61-90 box)"]
x = np.arange(len(x_labels))
width = 0.2
  
# DATA ENTRY  

# --- GUROBI DATA ---
g_times_c1, g_times_c2, g_times_c3, g_times_c4 = [], [], [], []
g_map_c1, g_map_c2, g_map_c3, g_map_c4 = [], [], [], []

# --- SIMULATED ANNEALING (SA) DATA ---
sa_times_c1, sa_times_c2, sa_times_c3, sa_times_c4 = [], [], [], []
sa_map_c1, sa_map_c2, sa_map_c3, sa_map_c4 = [], [], [], []

# --- QUANTUM ANNEALER (QA) DATA ---
qa_times_c1, qa_times_c2, qa_times_c3, qa_times_c4 = [], [], [], []
qa_map_c1, qa_map_c2, qa_map_c3, qa_map_c4 = [], [], [], []
  
# GENERATION FUNCTIONS
def create_plots(method_name, file_prefix, times, maps):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels = ['Case 1 (IoU)', 'Case 2 (IoU+IoM)', 'Case 3 (Sp)', 'Case 4 (IoU+IoM+Sp)']

    # 1. TIME PLOT
    plt.figure(figsize=(12, 6))
    for i in range(4):
        plt.bar(x + (i - 1.5) * width, times[i], width, label=labels[i], color=colors[i])
    plt.xlabel('Images (Sorted by Number of boxes)', fontsize=12)
    plt.ylabel('Execution Time (seconds)', fontsize=12)
    plt.title(f'{method_name}: Time Scalability vs Number of boxes', fontsize=14, fontweight='bold')
    plt.xticks(x, x_labels, fontsize=10)
    plt.ylim(0, 1.1) 
    plt.yticks(np.arange(0, 1.2, 0.1))
    plt.legend(); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
    plt.savefig(f'scalability_times_{file_prefix}_291img.png', dpi=300); plt.close()

    # 2. mAP PLOT
    plt.figure(figsize=(12, 6))
    for i in range(4):
        plt.bar(x + (i - 1.5) * width, maps[i], width, label=labels[i], color=colors[i])
    plt.xlabel('Images (Sorted by Number of boxes)', fontsize=12)
    plt.ylabel('mAP (Standard)', fontsize=12)
    plt.title(f'{method_name}: mAP Scalability vs Number of boxes', fontsize=14, fontweight='bold')
    plt.xticks(x, x_labels, fontsize=10)
    plt.ylim(0, 0.6)  
    plt.legend(); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
    plt.savefig(f'scalability_map_{file_prefix}_291img.png', dpi=300); plt.close()
  
# RUN PLOTS  

# Plot Gurobi
create_plots("Gurobi", "gurobi",
             [g_times_c1, g_times_c2, g_times_c3, g_times_c4],
             [g_map_c1, g_map_c2, g_map_c3, g_map_c4])

# Plot SA
create_plots("Simulated Annealing", "sa",
             [sa_times_c1, sa_times_c2, sa_times_c3, sa_times_c4],
             [sa_map_c1, sa_map_c2, sa_map_c3, sa_map_c4])

# Plot QA 1000
create_plots("Quantum Annealer (900 Reads)", "qa",
             [qa_times_c1, qa_times_c2, qa_times_c3, qa_times_c4],
             [qa_map_c1, qa_map_c2, qa_map_c3, qa_map_c4])

print("Plots are saved")
