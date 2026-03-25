import matplotlib.pyplot as plt
import numpy as np
import os

# INITIALIZATION & X-AXIS SETUP
image_IDs = [532481, 458755, 147740, 57597, 172946, 214539]
box_counts = [5, 23, 42, 62, 87, 99]

# Labels: "ID 532481\n(5 box)"
x_labels = [f"ID {img_id}\n({n} box)" for img_id, n in zip(image_IDs, box_counts)]

x = np.arange(len(x_labels))
width = 0.18

# DATA ENTRY 

# --- GUROBI DATA ---
g_times_c1, g_times_c2, g_times_c3, g_times_c4 = [], [], [], []
g_map_c1, g_map_c2, g_map_c3, g_map_c4 = [], [], [], []
g_f1_c1, g_f1_c2, g_f1_c3, g_f1_c4 = [], [], [], []
g_mae_c1, g_mae_c2, g_mae_c3, g_mae_c4 = [], [], [], []

# --- SIMULATED ANNEALING (SA) DATA ---
sa_times_c1, sa_times_c2, sa_times_c3, sa_times_c4 = [], [], [], []
sa_map_c1, sa_map_c2, sa_map_c3, sa_map_c4 = [], [], [], []
sa_f1_c1, sa_f1_c2, sa_f1_c3, sa_f1_c4 = [], [], [], []
sa_mae_c1, sa_mae_c2, sa_mae_c3, sa_mae_c4 = [], [], [], []

# --- QUANTUM ANNEALER (QA) 1000 READS DATA ---
qa_times_c1, qa_times_c2, qa_times_c3, qa_times_c4 = [], [], [], []
qa_map_c1, qa_map_c2, qa_map_c3, qa_map_c4 = [], [], [], []
qa_f1_c1, qa_f1_c2, qa_f1_c3, qa_f1_c4 = [], [], [], []
qa_mae_c1, qa_mae_c2, qa_mae_c3, qa_mae_c4 = [], [], [], []

# --- QUANTUM ANNEALER (QA) 3000 READS DATA ---
qa3k_times_c1, qa3k_times_c2, qa3k_times_c3, qa3k_times_c4 = [], [], [], []
qa3k_map_c1, qa3k_map_c2, qa3k_map_c3, qa3k_map_c4 = [], [], [], []
qa3k_f1_c1, qa3k_f1_c2, qa3k_f1_c3, qa3k_f1_c4 = [], [], [], []
qa3k_mae_c1, qa3k_mae_c2, qa3k_mae_c3, qa3k_mae_c4 = [], [], [], []

# --- QUANTUM ANNEALER (QA) 500 READS DATA ---
qa500_times_c1, qa500_times_c2, qa500_times_c3, qa500_times_c4 = [], [], [], []
qa500_map_c1, qa500_map_c2, qa500_map_c3, qa500_map_c4 = [], [], [], []
qa500_f1_c1, qa500_f1_c2, qa500_f1_c3, qa500_f1_c4 = [], [], [], []
qa500_mae_c1, qa500_mae_c2, qa500_mae_c3, qa500_mae_c4 = [], [], [], []

# GENERATION FUNCTIONS
def create_plots(method_name, file_prefix, times, maps, f1s, maes):
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
    plt.ylim(0, 2.2) 
    plt.yticks(np.arange(0, 2.4, 0.2))
    plt.legend(); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
    plt.savefig(f'scalability_times_{file_prefix}.png', dpi=300); plt.close()

    # 2. mAP PLOT
    plt.figure(figsize=(12, 6))
    for i in range(4):
        plt.bar(x + (i - 1.5) * width, maps[i], width, label=labels[i], color=colors[i])
    plt.xlabel('Images (Sorted by Number of boxes)', fontsize=12)
    plt.ylabel('mAP (Standard)', fontsize=12)
    plt.title(f'{method_name}: mAP Scalability vs Number of boxes', fontsize=14, fontweight='bold')
    plt.xticks(x, x_labels, fontsize=10)
    plt.ylim(0, 0.8)  
    plt.legend(); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
    plt.savefig(f'scalability_map_{file_prefix}.png', dpi=300); plt.close()

    # 3. F1-SCORE PLOT
    plt.figure(figsize=(12, 6))
    for i in range(4):
        plt.bar(x + (i - 1.5) * width, f1s[i], width, label=labels[i], color=colors[i])
    plt.xlabel('Images (Sorted by Number of boxes)', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.title(f'{method_name}: F1-Score Scalability vs Number of boxes', fontsize=14, fontweight='bold')
    plt.xticks(x, x_labels, fontsize=10)
    plt.ylim(0, 0.8) 
    plt.legend(); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
    plt.savefig(f'scalability_f1_{file_prefix}.png', dpi=300); plt.close()

    # 4. MAE PLOT
    plt.figure(figsize=(12, 6))
    for i in range(4):
        plt.bar(x + (i - 1.5) * width, maes[i], width, label=labels[i], color=colors[i])
    plt.xlabel('Images (Sorted by Number of boxes)', fontsize=12)
    plt.ylabel('MAE (Absolute Error)', fontsize=12)
    plt.title(f'{method_name}: MAE Scalability vs Number of boxes', fontsize=14, fontweight='bold')
    plt.xticks(x, x_labels, fontsize=10)
    plt.yticks(np.arange(0, 11, 1))
        
    plt.legend(); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
    plt.savefig(f'scalability_mae_{file_prefix}.png', dpi=300); plt.close()

 
# 3. RUN PLOTS
# Plot Gurobi
create_plots("Gurobi", "gurobi",
             [g_times_c1, g_times_c2, g_times_c3, g_times_c4],
             [g_map_c1, g_map_c2, g_map_c3, g_map_c4],
             [g_f1_c1, g_f1_c2, g_f1_c3, g_f1_c4],
             [g_mae_c1, g_mae_c2, g_mae_c3, g_mae_c4])

# Plot SA
create_plots("Simulated Annealing", "sa",
             [sa_times_c1, sa_times_c2, sa_times_c3, sa_times_c4],
             [sa_map_c1, sa_map_c2, sa_map_c3, sa_map_c4],
             [sa_f1_c1, sa_f1_c2, sa_f1_c3, sa_f1_c4],
             [sa_mae_c1, sa_mae_c2, sa_mae_c3, sa_mae_c4])

# Plot QA 1000
create_plots("Quantum Annealer (1000 Reads)", "qa_1000",
             [qa_times_c1, qa_times_c2, qa_times_c3, qa_times_c4],
             [qa_map_c1, qa_map_c2, qa_map_c3, qa_map_c4],
             [qa_f1_c1, qa_f1_c2, qa_f1_c3, qa_f1_c4],
             [qa_mae_c1, qa_mae_c2, qa_mae_c3, qa_mae_c4])

# Plot QA 3000
create_plots("Quantum Annealer (3000 Reads)", "qa_3000",
             [qa3k_times_c1, qa3k_times_c2, qa3k_times_c3, qa3k_times_c4],
             [qa3k_map_c1, qa3k_map_c2, qa3k_map_c3, qa3k_map_c4],
             [qa3k_f1_c1, qa3k_f1_c2, qa3k_f1_c3, qa3k_f1_c4],
             [qa3k_mae_c1, qa3k_mae_c2, qa3k_mae_c3, qa3k_mae_c4])

# Plot QA 500
create_plots("Quantum Annealer (500 Reads)", "qa_500",
             [qa500_times_c1, qa500_times_c2, qa500_times_c3, qa500_times_c4],
             [qa500_map_c1, qa500_map_c2, qa500_map_c3, qa500_map_c4],
             [qa500_f1_c1, qa500_f1_c2, qa500_f1_c3, qa500_f1_c4],
             [qa500_mae_c1, qa500_mae_c2, qa500_mae_c3, qa500_mae_c4])

print("Plots are saved")
