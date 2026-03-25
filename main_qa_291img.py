# IMPORT + INITIALIZATION
from dwave.system import DWaveSampler, EmbeddingComposite
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import time
import sys
from prettytable import PrettyTable
import json
import matplotlib.pyplot as plt

# SOLUTION TO KERNEL CRASH: ignore conflicts between OpenMP libraries (Gurobi vs. PyTorch/NumPy)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import build_qubo_matrix 
import build_qubo_matrix2
import build_qubo_matrix3
import build_qubo_matrix4
import metrics
import RCNN
import logging

# Silenced torchvision/pytorch warnings
logging.getLogger("torchvision").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# Path to the file containing the ground truths for each image
instances_file = './coco2017/annotations/instances_val2017.json'
coco = COCO(instances_file) # Initialization

# ========================================================
# DATA EXTRACTION AND FILTERING (300 images subset)
# ========================================================
# Select only images that have people
catIds = coco.getCatIds(catNms=['person'])
image_IDs = coco.getImgIds(catIds=catIds) 

# We reduce the analysis to 300 images
if len(image_IDs) > 300:
    image_IDs = image_IDs[:300] 

# DEVICE SETUP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# SAMPLER INITIALIZATION
sampler = EmbeddingComposite(DWaveSampler()) 

num_reads = 900

# LOADING MODEL
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT 
model = fasterrcnn_resnet50_fpn(weights=weights)

# NMS parameters (let's keep all boxes for QUBO problem)
model.roi_heads.nms_thresh = 0.9  
model.roi_heads.score_thresh = 0.6 

model.to(device)
model.eval() 

print("INITIALIZATION COMPLETED")

# BOUNDING BOX ACQUISITION WITH GPU
gpu_data = []
valid_count = 0

print("\n--- STARTING RCNN EXTRACTION AND FILTERING (2-90 boxes) ---")
for img_id in image_IDs:
    
    img_info = coco.loadImgs(img_id)[0]
    file_name = img_info['file_name']
    image_path = f"./coco2017/val2017/{file_name}"

    t_rcnn_start = time.perf_counter()
    raw_boxes, scores, original_image = RCNN.faster_rcnn(image_path, model, device) 
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_rcnn_end = time.perf_counter()

    num_boxes = len(raw_boxes)

    # Filter: keep images with 2 to 90 boxes
    if num_boxes < 2 or num_boxes > 90:
        continue
    
    valid_count += 1
    boxes_xywh = []
    for box in raw_boxes:
        x1, y1, x2, y2 = box
        w = x2 - x1  
        h = y2 - y1  
        boxes_xywh.append([x1, y1, w, h]) 
    
    print(f"[{valid_count}] Image ID: {img_id} | File: {file_name} | Boxes found: {len(raw_boxes)} | RCNN Time: {t_rcnn_end - t_rcnn_start:.3f}s")

    gpu_data.append({
        'image_id': img_id,
        'boxes': np.array(boxes_xywh),
        'scores': scores,
        'file_name': file_name
    })

print(f"\nExtracted and filtered {valid_count} valid images!")

# Sort by number of bounding boxes before slicing
gpu_data.sort(key=lambda x: len(x['boxes']))
valid_image_IDs = [data['image_id'] for data in gpu_data]

# Compute slicing indices for Density categories
idx_cut_low = 0
idx_cut_med = 0

for i, data in enumerate(gpu_data):
    n_boxes = len(data['boxes'])
    if n_boxes <= 20:
        idx_cut_low = i + 1 
    if n_boxes <= 60:
        idx_cut_med = i + 1

# Ground truths extraction
ground_truths = {}
for data in gpu_data:
    image_id = data['image_id']
    file_name = data['file_name']

    annIds = coco.getAnnIds(imgIds=image_id, catIds=[1])
    anns = coco.loadAnns(annIds) 
    
    gt_boxes = []
    gt_labels = []
    for ann in anns:
        gt_boxes.append(ann['bbox']) 
        gt_labels.append(ann['category_id']) 
    
    ground_truths[image_id] = {
        'boxes': np.array(gt_boxes),
        'labels': np.array(gt_labels),
        'file_name': file_name
    }

print("GROUND TRUTHS AND PREDICTIONS LOADING COMPLETED")
print(f"\nTHRESHOLDS IDENTIFIED ON {len(gpu_data)} IMAGES:")
print(f"- Low Density (2-20 box): 0 to {idx_cut_low} ({idx_cut_low} images)")
print(f"- Med Density (21-60 box): {idx_cut_low} to {idx_cut_med} ({idx_cut_med - idx_cut_low} images)")
print(f"- High Density (>60 box): {idx_cut_med} to end ({len(gpu_data) - idx_cut_med} images)")

# HELPER FUNCTIONS
def conversion_Q(Q):
    """Converts the QUBO matrix into triangular minimization form for D-Wave"""
    Q_dict = {}
    n = len(Q)
    for i in range(n):
        if Q[i, i] != 0:
            Q_dict[(i, i)] = -Q[i, i]
        for j in range(i + 1, n):
            val = -2 * Q[i, j]
            if val != 0: 
                Q_dict[(i, j)] = val
    return Q_dict

def process_qa_solution(best_sample, boxes, scores, image_id):
    """Extracts bounding boxes from the best solution obtained by QA"""
    N = len(boxes)
    sol_qa = np.zeros(N, dtype=int)
    for idx, val in best_sample.items():
        sol_qa[idx] = val

    kept_indices = np.where(sol_qa == 1)[0]
    image_predictions = []
    if len(kept_indices) > 0:
        kept_boxes = boxes[kept_indices]
        kept_scores = scores[kept_indices]
        for k in range(len(kept_boxes)):
            prediction = {
                "image_id": int(image_id),
                "category_id": 1,
                "bbox": kept_boxes[k].tolist(),
                "score": float(kept_scores[k])
            }
            image_predictions.append(prediction)
    return image_predictions

def extract_top_10(sampleset, N):
    """Extracts the top 10 solutions found by Quantum Annealer"""
    top_10_list = []
    limit = min(10, len(sampleset))
    for i, datum in enumerate(sampleset.data(['sample', 'energy', 'num_occurrences'])):
        if i >= limit: break
        sol_vector = [0] * N
        for idx, val in datum.sample.items():
            sol_vector[idx] = int(val)
        top_10_list.append({
            "rank": i + 1,
            "vector": sol_vector,
            "energy": float(datum.energy),
            "occurrences": int(datum.num_occurrences)
        })
    return top_10_list

# ========================================================
# QUANTUM ANNEALING RUN
# ========================================================

best_alpha_1, best_alpha_2, best_alpha_3, best_alpha_4 = 0.58, 0.60, 0.60, 0.62

qa_times_case1, qa_times_case2, qa_times_case3, qa_times_case4 = {}, {}, {}, {}
qa_results_case1, qa_results_case2, qa_results_case3, qa_results_case4 = [], [], [], []
counts_case1, counts_case2, counts_case3, counts_case4 = {}, {}, {}, {}
top10_imgs_case1, top10_imgs_case2, top10_imgs_case3, top10_imgs_case4 = [], [], [], []

output_dir = "qa_top10_results_291imm"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("\n--- STARTING QPU CONNECTION (D-WAVE) ---")
for i, data in enumerate(gpu_data):
    boxes = data['boxes'].astype(np.float64)
    scores = data['scores'].astype(np.float64)
    image_id = data['image_id']
    gt_count = len(ground_truths[image_id]['boxes'])
    N = len(boxes) 

    # CASE 1
    L, P = build_qubo_matrix.qubo_matrices(boxes, scores)
    Q1 = np.round(best_alpha_1 * L - (1 - best_alpha_1) * P, 6)
    sampleset = sampler.sample_qubo(conversion_Q(Q1), num_reads=num_reads)
    qa_times_case1[image_id] = sampleset.info['timing']['qpu_access_time'] / 1000000.0
    top10_imgs_case1.append({"image_id": int(image_id), "solutions": extract_top_10(sampleset, N)})
    preds = process_qa_solution(sampleset.first.sample, boxes, scores, image_id)
    qa_results_case1.extend(preds)
    counts_case1[image_id] = (len(preds), gt_count)

    # CASE 2
    L, P = build_qubo_matrix2.qubo_matrices(boxes, scores)
    Q2 = np.round(best_alpha_2 * L - (1 - best_alpha_2) * P, 6)
    sampleset = sampler.sample_qubo(conversion_Q(Q2), num_reads=num_reads)
    qa_times_case2[image_id] = sampleset.info['timing']['qpu_access_time'] / 1000000.0
    top10_imgs_case2.append({"image_id": int(image_id), "solutions": extract_top_10(sampleset, N)})
    preds = process_qa_solution(sampleset.first.sample, boxes, scores, image_id)
    qa_results_case2.extend(preds)
    counts_case2[image_id] = (len(preds), gt_count)

    # CASE 3
    L, P1, P2 = build_qubo_matrix3.qubo_matrices(boxes, scores)
    beta = (1 - best_alpha_3) / 2
    Q3 = np.round(best_alpha_3 * L - beta * P1 - beta * P2, 6)
    sampleset = sampler.sample_qubo(conversion_Q(Q3), num_reads=num_reads)
    qa_times_case3[image_id] = sampleset.info['timing']['qpu_access_time'] / 1000000.0
    top10_imgs_case3.append({"image_id": int(image_id), "solutions": extract_top_10(sampleset, N)})
    preds = process_qa_solution(sampleset.first.sample, boxes, scores, image_id)
    qa_results_case3.extend(preds)
    counts_case3[image_id] = (len(preds), gt_count)

    # CASE 4
    L, P1, P2 = build_qubo_matrix4.qubo_matrices(boxes, scores)
    beta = (1 - best_alpha_4) / 2
    Q4 = np.round(best_alpha_4 * L - beta * P1 - beta * P2, 6)
    sampleset = sampler.sample_qubo(conversion_Q(Q4), num_reads=num_reads)
    qa_times_case4[image_id] = sampleset.info['timing']['qpu_access_time'] / 1000000.0
    top10_imgs_case4.append({"image_id": int(image_id), "solutions": extract_top_10(sampleset, N)})
    preds = process_qa_solution(sampleset.first.sample, boxes, scores, image_id)
    qa_results_case4.extend(preds)
    counts_case4[image_id] = (len(preds), gt_count)

    print(f"Quantum Annealer: Processed {i + 1} / {valid_count} images (Image ID: {image_id}, Boxes: {N})")

    # INCREMENTAL JSON SAVING
    with open(f"{output_dir}/top10_case1.json", 'w') as f: json.dump(top10_imgs_case1, f, indent=4)
    with open(f"{output_dir}/top10_case2.json", 'w') as f: json.dump(top10_imgs_case2, f, indent=4)
    with open(f"{output_dir}/top10_case3.json", 'w') as f: json.dump(top10_imgs_case3, f, indent=4)
    with open(f"{output_dir}/top10_case4.json", 'w') as f: json.dump(top10_imgs_case4, f, indent=4)


# ========================================================
# DENSITY AGGREGATION AND PLOTTING
# ========================================================
ids_low  = valid_image_IDs[0 : idx_cut_low]
ids_med  = valid_image_IDs[idx_cut_low : idx_cut_med]
ids_high = valid_image_IDs[idx_cut_med : ]

subsets = [
    ("Low (2-20)", ids_low),
    ("Med (21-60)", ids_med),
    ("High (>60)", ids_high),
    ("All", valid_image_IDs)
]

def get_avg_time(time_dict, subset_ids):
    times = [time_dict[i] for i in subset_ids if i in time_dict]
    return sum(times) / len(times) if times else 0.0

table_time = PrettyTable()
table_time.title = "Quantum Annealer - Times by Density - 291 images"
table_time.field_names = ["Case", "Low Avg (s)", "Med Avg (s)", "High Avg (s)", "Total Sum (s)"]

avg_times_plot = {1: [], 2: [], 3: [], 4: []}

for case_num, t_dict in [(1, qa_times_case1), (2, qa_times_case2), (3, qa_times_case3), (4, qa_times_case4)]:
    low_t = get_avg_time(t_dict, subsets[0][1])
    med_t = get_avg_time(t_dict, subsets[1][1])
    high_t = get_avg_time(t_dict, subsets[2][1])
    total_t = sum(t_dict.values())
    
    table_time.add_row([f"Case {case_num}", f"{low_t:.4f}", f"{med_t:.4f}", f"{high_t:.4f}", f"{total_t:.4f}"])
    avg_times_plot[case_num].extend([low_t, med_t, high_t])

print("\n")
print(table_time)
print("\n")

cases_info = [
    ('Case 1', qa_results_case1, counts_case1),
    ('Case 2', qa_results_case2, counts_case2),
    ('Case 3', qa_results_case3, counts_case3),
    ('Case 4', qa_results_case4, counts_case4)
]

table_metrics = PrettyTable()
table_metrics.title = "Quantum Annealer - Metrics by Density - 291 images"
table_metrics.field_names = ["Case", "Density", "Prec", "Rec", "F1", "mAP(std)", "mAP(.50)", "mAR(10)", "mAR(100)", "MAE", "RMSE"]

plot_data = {'map': {1:[], 2:[], 3:[], 4:[]}, 'f1': {1:[], 2:[], 3:[], 4:[]}, 'mae': {1:[], 2:[], 3:[], 4:[]}}
original_stdout = sys.stdout

for case_idx, (name, preds, counts_dict) in enumerate(cases_info, start=1):
    coco_dt = coco.loadRes(preds) if len(preds) > 0 else None
    
    for sub_name, sub_ids in subsets:
        diffs = [counts_dict[img_id][0] - counts_dict[img_id][1] for img_id in sub_ids if img_id in counts_dict]
        if diffs:
            mae = np.mean(np.abs(diffs))
            rmse = np.sqrt(np.mean(np.square(diffs)))
        else:
            mae = rmse = 0.0

        if coco_dt is None or len(sub_ids) == 0:
            table_metrics.add_row([name, sub_name, "0", "0", "0", "0", "0", "0", "0", f"{mae:.4f}", f"{rmse:.4f}"])
            if sub_name != "All": # BUG FIXED HERE
                plot_data['map'][case_idx].append(0.0)
                plot_data['f1'][case_idx].append(0.0)
                plot_data['mae'][case_idx].append(mae)
            continue
            
        coco_eval = COCOeval(coco, coco_dt, 'bbox')
        coco_eval.params.imgIds = sub_ids
        coco_eval.params.catIds = [1]
        
        sys.stdout = open(os.devnull, 'w')
        try:
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            mstd, m50 = coco_eval.stats[0], coco_eval.stats[1]
            m10, m100 = coco_eval.stats[7], coco_eval.stats[8]
        except:
            mstd = m50 = m10 = m100 = 0.0
        finally:
            sys.stdout = original_stdout

        p, r, f = metrics.compute_metrics(coco, preds, sub_ids)
        table_metrics.add_row([name, sub_name, f"{p:.4f}", f"{r:.4f}", f"{f:.4f}", f"{mstd:.4f}", f"{m50:.4f}", f"{m10:.4f}", f"{m100:.4f}", f"{mae:.4f}", f"{rmse:.4f}"])
        
        if sub_name != "All": # BUG FIXED HERE
            plot_data['map'][case_idx].append(mstd)
            plot_data['f1'][case_idx].append(f)
            plot_data['mae'][case_idx].append(mae)
            
    table_metrics.add_row(["-"*6, "-"*12, "-"*6, "-"*6, "-"*6, "-"*8, "-"*8, "-"*7, "-"*8, "-"*6, "-"*6])

print(table_metrics)
print("\n")


# ========================================================
# PLOTS (Time, mAP, F1, MAE)
# ========================================================
x_labels = ["Low\n(2-20 box)", "Medium\n(21-60 box)", "High\n(61-90 box)"]
x = np.arange(len(x_labels))
width = 0.2
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# 1. TIME PLOT
plt.figure(figsize=(12, 6))
for i, color in zip(range(1, 5), colors):
    offset = (i - 2.5) * width
    label = f'Case {i}'
    if i==1: label += ' (IoU)'
    if i==2: label += ' (IoU+IoM)'
    if i==3: label += ' (Sp)'
    if i==4: label += ' (IoU+IoM+Sp)'
    plt.bar(x + offset, avg_times_plot[i], width, label=label, color=color)

plt.xlabel('Density Categories', fontsize=12)
plt.ylabel('Average QPU Access Time (s)', fontsize=12)
plt.title('Quantum Annealer: Time Scalability vs Complexity with 291 images', fontsize=14, fontweight='bold')
plt.xticks(x, x_labels, fontsize=10)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('scalability_times_qa_291img.png', dpi=300)
plt.close()

# 2. mAP PLOT
plt.figure(figsize=(12, 6))
for i, color in zip(range(1, 5), colors):
    label = f'Case {i}'
    if i==1: label += ' (IoU)'
    if i==2: label += ' (IoU+IoM)'
    if i==3: label += ' (Sp)'
    if i==4: label += ' (IoU+IoM+Sp)'
    plt.bar(x + (i - 2.5) * width, plot_data['map'][i], width, label=label, color=color)
plt.xlabel('Density Categories', fontsize=12)
plt.ylabel('mAP (Standard)', fontsize=12)
plt.title('Quantum Annealer: mAP Scalability vs Complexity with 291 images', fontsize=14, fontweight='bold')
plt.xticks(x, x_labels, fontsize=10)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('scalability_map_qa_291img.png', dpi=300)
plt.close()

# 3. F1 PLOT
plt.figure(figsize=(12, 6))
for i, color in zip(range(1, 5), colors):
    label = f'Case {i}'
    if i==1: label += ' (IoU)'
    if i==2: label += ' (IoU+IoM)'
    if i==3: label += ' (Sp)'
    if i==4: label += ' (IoU+IoM+Sp)'
    plt.bar(x + (i - 2.5) * width, plot_data['f1'][i], width, label=label, color=color)
plt.xlabel('Density Categories', fontsize=12)
plt.ylabel('F1-Score', fontsize=12)
plt.title('Quantum Annealer: F1-Score Scalability vs Complexity with 291 images', fontsize=14, fontweight='bold')
plt.xticks(x, x_labels, fontsize=10)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('scalability_f1_qa_291img.png', dpi=300)
plt.close()

# 4. MAE PLOT
plt.figure(figsize=(12, 6))
for i, color in zip(range(1, 5), colors):
    label = f'Case {i}'
    if i==1: label += ' (IoU)'
    if i==2: label += ' (IoU+IoM)'
    if i==3: label += ' (Sp)'
    if i==4: label += ' (IoU+IoM+Sp)'
    plt.bar(x + (i - 2.5) * width, plot_data['mae'][i], width, label=label, color=color)
plt.xlabel('Density Categories', fontsize=12)
plt.ylabel('MAE (Average Absolute Error)', fontsize=12)
plt.title('Quantum Annealer: MAE Scalability vs Complexity with 291 images', fontsize=14, fontweight='bold')
plt.xticks(x, x_labels, fontsize=10)
plt.yticks(np.arange(0, 11, 1))
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('scalability_mae_qa_291img.png', dpi=300)
plt.close()