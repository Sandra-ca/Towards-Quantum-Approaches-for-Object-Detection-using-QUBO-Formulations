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
# List for scalability (6 specific images)
image_IDs = [532481, 458755, 147740, 57597, 172946, 214539] # 5, 23, 42, 62, 87, 99 box
# ========================================================

# DEVICE SETUP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# SAMPLER INITIALIZATION
# Default quantum sampler
sampler = EmbeddingComposite(DWaveSampler()) 

num_reads = 500

# LOADING MODEL
# COCO weights
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT  # DEFAULT means that it must use skills already developed with COCO's train 
model = fasterrcnn_resnet50_fpn(weights=weights)

# NMS parameters (let's keep all boxes for QUBO problem)
model.roi_heads.nms_thresh = 0.9  # Raise the threshold to 0.9 to keep overlaps
model.roi_heads.score_thresh = 0.6 # Minimum confidence score: 60%

model.to(device)
model.eval() # Evaluation mode

print("INITIALIZATION COMPLETED")

# BOUNDING BOX ACQUISITION WITH GPU

# Data list initialization
gpu_data = []
valid_count = 0

for i, img_id in enumerate(image_IDs):
    
    # Load image data
    img_info = coco.loadImgs(img_id)[0]
    file_name = img_info['file_name']

    # BOUNDING BOXES
    image_path = f"./coco2017/val2017/{file_name}"

    t_rcnn_start = time.perf_counter()
    raw_boxes, scores, original_image = RCNN.faster_rcnn(image_path, model, device) 
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_rcnn_end = time.perf_counter()

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

# Sort by number of bounding boxes
gpu_data.sort(key=lambda x: len(x['boxes']))
valid_image_IDs = [data['image_id'] for data in gpu_data]

# Ground truths extraction
ground_truths = {}
for i, data in enumerate(gpu_data):
    image_id = data['image_id']
    boxes = data['boxes']
    scores = data['scores']
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

# HELPER FUNCTIONS
def conversion_Q(Q):
    """Converts the QUBO matrix into triangular minimization form for D-Wave"""
    Q_dict = {}
    n = len(Q)

    for i in range(n):
        # Diagonal terms
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
        if i >= limit:
            break
            
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

# Best alpha computed previously
best_alpha_1 = 0.58
best_alpha_2 = 0.60
best_alpha_3 = 0.60
best_alpha_4 = 0.62

# Lists for execution times
qa_times_case1, qa_times_case2, qa_times_case3, qa_times_case4 = [], [], [], []

# Lists for predictions
qa_results_case1, qa_results_case2, qa_results_case3, qa_results_case4 = [], [], [], []

# Lists for top 10 solutions
top10_imgs_case1, top10_imgs_case2, top10_imgs_case3, top10_imgs_case4 = [], [], [], []

# Lists for pred_count and gt_count (for MAE and RMSE)
counts_case1, counts_case2, counts_case3, counts_case4 = [], [], [], []

# We analyze every image only once
for i, data in enumerate(gpu_data):
    boxes = data['boxes']
    scores = data['scores']
    image_id = data['image_id']
    
    # Ground Truth count for this image
    gt_count = len(ground_truths[image_id]['boxes'])
    
    boxes = boxes.astype(np.float64)
    scores = scores.astype(np.float64)
    N = len(boxes) 

    # --- CASE 1 (IoU) ---
    L, P = build_qubo_matrix.qubo_matrices(boxes, scores)
    Q1 = best_alpha_1 * L - (1 - best_alpha_1) * P
    Q1 = np.round(Q1, decimals=6)
    Q_dict = conversion_Q(Q1)

    sampleset = sampler.sample_qubo(Q_dict, num_reads=num_reads)
    qpu_time_sec = sampleset.info['timing']['qpu_access_time'] / 1000000.0
    qa_times_case1.append(qpu_time_sec)

    top_10_case1 = extract_top_10(sampleset, N)
    top10_imgs_case1.append({"image_id": int(image_id), "solutions": top_10_case1})

    best_solution_1 = sampleset.first.sample
    preds_case1 = process_qa_solution(best_solution_1, boxes, scores, image_id)
    qa_results_case1.extend(preds_case1)
    
    counts_case1.append((len(preds_case1), gt_count))

    # chain break best sol
    best_cbf = sampleset.record['chain_break_fraction'][0]

    # mean chain break
    mean_cbf = np.average(sampleset.record['chain_break_fraction'], weights=sampleset.record['num_occurrences'])

    print(f"Img ID {image_id} Case 1 - CBF Best Sol: {best_cbf * 100:.2f}% | Mean CBF (500 reads): {mean_cbf * 100:.2f}%")

    # --- CASE 2 (IoU + IoM) ---
    L, P = build_qubo_matrix2.qubo_matrices(boxes, scores)
    Q2 = best_alpha_2 * L - (1 - best_alpha_2) * P
    Q2 = np.round(Q2, decimals=6)
    Q_dict = conversion_Q(Q2)

    sampleset = sampler.sample_qubo(Q_dict, num_reads=num_reads)
    qpu_time_sec = sampleset.info['timing']['qpu_access_time'] / 1000000.0
    qa_times_case2.append(qpu_time_sec)

    top_10_case2 = extract_top_10(sampleset, N)
    top10_imgs_case2.append({"image_id": int(image_id), "solutions": top_10_case2})

    best_solution_2 = sampleset.first.sample
    preds_case2 = process_qa_solution(best_solution_2, boxes, scores, image_id)
    qa_results_case2.extend(preds_case2)
    
    counts_case2.append((len(preds_case2), gt_count))

    # chain break best sol
    best_cbf = sampleset.record['chain_break_fraction'][0]

    # mean chain break
    mean_cbf = np.average(sampleset.record['chain_break_fraction'], weights=sampleset.record['num_occurrences'])

    print(f"Img ID {image_id} Case 2 - CBF Best Sol: {best_cbf * 100:.2f}% | Mean CBF (500 reads): {mean_cbf * 100:.2f}%")

    # --- CASE 3 (IoU + Sp) ---
    L, P1, P2 = build_qubo_matrix3.qubo_matrices(boxes, scores)
    beta = (1 - best_alpha_3) / 2
    Q3 = best_alpha_3 * L - beta * P1 - beta * P2
    Q3 = np.round(Q3, decimals=6)
    Q_dict = conversion_Q(Q3)

    sampleset = sampler.sample_qubo(Q_dict, num_reads=num_reads)
    qpu_time_sec = sampleset.info['timing']['qpu_access_time'] / 1000000.0
    qa_times_case3.append(qpu_time_sec)

    top_10_case3 = extract_top_10(sampleset, N)
    top10_imgs_case3.append({"image_id": int(image_id), "solutions": top_10_case3})

    best_solution_3 = sampleset.first.sample
    preds_case3 = process_qa_solution(best_solution_3, boxes, scores, image_id)
    qa_results_case3.extend(preds_case3)
    
    counts_case3.append((len(preds_case3), gt_count))

    # chain break best sol
    best_cbf = sampleset.record['chain_break_fraction'][0]

    # mean chain break
    mean_cbf = np.average(sampleset.record['chain_break_fraction'], weights=sampleset.record['num_occurrences'])

    print(f"Img ID {image_id} Case 3 - CBF Best Sol: {best_cbf * 100:.2f}% | Mean CBF (500 reads): {mean_cbf * 100:.2f}%")

    # --- CASE 4 (IoU+IoM + Sp) ---
    L, P1, P2 = build_qubo_matrix4.qubo_matrices(boxes, scores)
    beta = (1 - best_alpha_4) / 2
    Q4 = best_alpha_4 * L - beta * P1 - beta * P2
    Q4 = np.round(Q4, decimals=6)
    Q_dict = conversion_Q(Q4)

    sampleset = sampler.sample_qubo(Q_dict, num_reads=num_reads)
    qpu_time_sec = sampleset.info['timing']['qpu_access_time'] / 1000000.0
    qa_times_case4.append(qpu_time_sec)

    top_10_case4 = extract_top_10(sampleset, N)
    top10_imgs_case4.append({"image_id": int(image_id), "solutions": top_10_case4})

    best_solution_4 = sampleset.first.sample
    preds_case4 = process_qa_solution(best_solution_4, boxes, scores, image_id)
    qa_results_case4.extend(preds_case4)

    counts_case4.append((len(preds_case4), gt_count))

    # chain break best sol
    best_cbf = sampleset.record['chain_break_fraction'][0]

    # mean chain break
    mean_cbf = np.average(sampleset.record['chain_break_fraction'], weights=sampleset.record['num_occurrences'])
    
    print(f"Img ID {image_id} Case 4 - CBF Best Sol: {best_cbf * 100:.2f}% | Mean CBF (500 reads): {mean_cbf * 100:.2f}%")

# ========================================================
# SAVING JSON RESULTS
# ========================================================
output_dir = "qa_top10_results_6imm_500reads"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(f"{output_dir}/top10_case1.json", 'w') as f: json.dump(top10_imgs_case1, f, indent=4)
with open(f"{output_dir}/top10_case2.json", 'w') as f: json.dump(top10_imgs_case2, f, indent=4)
with open(f"{output_dir}/top10_case3.json", 'w') as f: json.dump(top10_imgs_case3, f, indent=4)
with open(f"{output_dir}/top10_case4.json", 'w') as f: json.dump(top10_imgs_case4, f, indent=4)

print("Saved 4 JSON files")   

# ========================================================
# TABLES
# ========================================================

# Prepare X-axis labels and data for plotting
img_ids = [data['image_id'] for data in gpu_data]
box_counts = [len(data['boxes']) for data in gpu_data]
x_labels = [f"ID {img_id}\n({n} box)" for img_id, n in zip(img_ids, box_counts)]
x = np.arange(len(img_ids))
width = 0.2 # width of the bars

# FINAL SUMMARY TIME TABLE PER IMAGE 
table_time = PrettyTable()
table_time.title = "Quantum Annealer- 6 imgs - Times per Image"
table_time.field_names = ["Image ID", "Num Boxes", "Case 1 (s)", "Case 2 (s)", "Case 3 (s)", "Case 4 (s)"]

for i in range(len(img_ids)):
    table_time.add_row([
        img_ids[i], 
        box_counts[i],
        f"{qa_times_case1[i]:.4f}", 
        f"{qa_times_case2[i]:.4f}", 
        f"{qa_times_case3[i]:.4f}", 
        f"{qa_times_case4[i]:.4f}"
    ])

print("\n")
print(table_time)
print("\n")

# FINAL SUMMARY METRICS TABLE PER IMAGE 
# We structure the subsets to evaluate each image individually
subsets = [(f"ID {img_ids[i]}\n({box_counts[i]} box)", [img_ids[i]], i) for i in range(len(img_ids))]

# Lists to dynamically store scores for plotting
map_scores_c1, f1_scores_c1, mae_scores_c1 = [], [], []
map_scores_c2, f1_scores_c2, mae_scores_c2 = [], [], []
map_scores_c3, f1_scores_c3, mae_scores_c3 = [], [], []
map_scores_c4, f1_scores_c4, mae_scores_c4 = [], [], []

cases_info = [
    ('Case 1', best_alpha_1, qa_results_case1, counts_case1, map_scores_c1, f1_scores_c1, mae_scores_c1),
    ('Case 2', best_alpha_2, qa_results_case2, counts_case2, map_scores_c2, f1_scores_c2, mae_scores_c2),
    ('Case 3', best_alpha_3, qa_results_case3, counts_case3, map_scores_c3, f1_scores_c3, mae_scores_c3),
    ('Case 4', best_alpha_4, qa_results_case4, counts_case4, map_scores_c4, f1_scores_c4, mae_scores_c4)
]

# Dictionary to hold results grouped by image to print them cleanly
results_dict = {i: [] for i in range(len(img_ids))}
original_stdout = sys.stdout

for name, alpha, preds, counts, map_list, f1_list, mae_list in cases_info:
    if len(preds) > 0:
        coco_dt = coco.loadRes(preds)
    else:
        coco_dt = None
    
    for i, (sub_name, sub_ids, idx) in enumerate(subsets):
        # Extract counts specific to this image
        p_c, g_c = counts[i]
        mae = abs(p_c - g_c)
        rmse = float(mae) # For a single image, RMSE is equal to MAE
        
        if coco_dt is None:
            results_dict[i].append((name, 0, 0, 0, 0, 0, 0, 0, mae, rmse))
            map_list.append(0.0)
            f1_list.append(0.0)
            mae_list.append(mae)
            continue
            
        coco_eval = COCOeval(coco, coco_dt, 'bbox')
        coco_eval.params.imgIds = sub_ids
        coco_eval.params.catIds = [1]
        
        sys.stdout = open(os.devnull, 'w')
        try:
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            mAP_std = coco_eval.stats[0]
            mAP_50 = coco_eval.stats[1]
            mAR_10 = coco_eval.stats[7]
            mAR_100 = coco_eval.stats[8]
        except Exception:
            mAP_std = mAP_50 = mAR_10 = mAR_100 = 0.0
        finally:
            sys.stdout = original_stdout

        precision, recall, f1 = metrics.compute_metrics(coco, preds, sub_ids)
        
        # Populate plotting arrays dynamically
        map_list.append(mAP_std)
        f1_list.append(f1)
        mae_list.append(mae)
        
        results_dict[i].append((name, precision, recall, f1, mAP_std, mAP_50, mAR_10, mAR_100, mae, rmse))

# Print Metrics Table grouped by Image
table_metrics = PrettyTable()
table_metrics.title = "Quantum Annealer- 6 imgs - Metrics per Image"
table_metrics.field_names = ["Image ID (Boxes)", "Case", "Prec", "Rec", "F1", "mAP(std)", "mAP(.50)", "mAR(10)", "mAR(100)", "MAE", "RMSE"]

for i, (sub_name, sub_ids, idx) in enumerate(subsets):
    for j, res in enumerate(results_dict[i]):
        name, p, r, f, mstd, m50, m10, m100, mae, rmse = res
        # Show image ID only on the first row of the group for readability
        col1 = sub_name if j == 0 else ""
        table_metrics.add_row([col1, name, f"{p:.4f}", f"{r:.4f}", f"{f:.4f}", f"{mstd:.4f}", f"{m50:.4f}", f"{m10:.4f}", f"{m100:.4f}", f"{mae:.4f}", f"{rmse:.4f}"])
    table_metrics.add_row(["-"*18, "-"*8, "-"*6, "-"*6, "-"*6, "-"*8, "-"*8, "-"*7, "-"*8, "-"*6, "-"*6])

print(table_metrics)
print("\n")
