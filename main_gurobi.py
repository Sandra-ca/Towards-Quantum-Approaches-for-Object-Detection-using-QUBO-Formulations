# IMPORT + INITIALIZATION
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import time
import sys
from prettytable import PrettyTable
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
import qubo_solver

# silenced torchvision/pytorch
logging.getLogger("torchvision").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# path to the file containing the ground truths for each image (called with an ID)
instances_file = './coco2017/annotations/instances_val2017.json'
coco = COCO(instances_file) # initialization

# ========================================================
# list for scalability (6 images)
image_IDs = [532481, 458755, 147740, 57597, 172946, 214539] # 5, 23, 42, 62, 87, 99 boxes
# ========================================================

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# LOADING MODEL
# COCO weights
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT  # DEFAULT means that it must use skills already developed with COCO's train 
model = fasterrcnn_resnet50_fpn(weights=weights)

# NMS parameters (let's keep all boxes for QUBO problem)
model.roi_heads.nms_thresh = 0.9  # raise the threshold to 0.9 to keep overlaps
model.roi_heads.score_thresh = 0.6 # minimum confidence score: 60%

model.to(device)
model.eval() # evaluation mode

print("INITIALIZATION COMPLETED")

# BOUNDING BOX ACQUISITION WITH GPU

# data list initialization
gpu_data = []
valid_count = 0

for i, img_id in enumerate(image_IDs):
    
    # load image data
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

gpu_data.sort(key=lambda x: len(x['boxes']))
valid_image_IDs = [data['image_id'] for data in gpu_data]

# ground truths extraction
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

print("GT AND PREDICTIONS LOADING IS COMPLETED")

# ========================================================
# GUROBI SCALABILITY RUN
# ========================================================

# best alpha computed using best_alpha_gurobi.py
best_alpha_1 = 0.58
best_alpha_2 = 0.60
best_alpha_3 = 0.60
best_alpha_4 = 0.62

# lists for times
gurobi_times_case1 = []
gurobi_times_case2 = []
gurobi_times_case3 = []
gurobi_times_case4 = []

# lists for predictions
gurobi_results_case1 = []
gurobi_results_case2 = []
gurobi_results_case3 = []
gurobi_results_case4 = []

# lists for pred_count and gt_count for MAE and RMSE
counts_case1 = []
counts_case2 = []
counts_case3 = []
counts_case4 = []

# GUROBI WARM-UP (DUMMY RUN)
Q = np.array([[1.0, -2.0], [-2.0, 1.0]])
_ = qubo_solver.qubo(Q)

# we analyze every image only once
for i, data in enumerate(gpu_data):
    boxes = data['boxes']
    scores = data['scores']
    image_id = data['image_id']
    
    # GT of this img
    gt_count = len(ground_truths[image_id]['boxes'])
    
    boxes = boxes.astype(np.float64)
    scores = scores.astype(np.float64)
    N = len(boxes) 

    print(f"\n--- Processing Image ID {image_id} ({N} boxes) ---")

    # --- CASE 1 (IoU) ---
    L, P = build_qubo_matrix.qubo_matrices(boxes, scores)
    Q1 = best_alpha_1 * L - (1 - best_alpha_1) * P
    Q1 = np.round(Q1, decimals=6)
    
    t_gurobi_start = time.perf_counter()
    sol_gurobi = qubo_solver.qubo(Q1)
    duration_gurobi = time.perf_counter() - t_gurobi_start
    gurobi_times_case1.append(duration_gurobi)

    print(f"Case 1 Sol: {sol_gurobi.tolist()}")

    sol_gurobi = np.array(sol_gurobi)
    kept_indices = np.where(sol_gurobi == 1)[0]
    counts_case1.append((len(kept_indices), gt_count))

    image_predictions = []
    if len(kept_indices) > 0:
        kept_boxes = boxes[kept_indices]
        kept_scores = scores[kept_indices]
        for k in range(len(kept_boxes)):
            image_predictions.append({
                "image_id": int(image_id),
                "category_id": 1, 
                "bbox": kept_boxes[k].tolist(), 
                "score": float(kept_scores[k])
            })
    gurobi_results_case1.extend(image_predictions)

    # --- CASE 2 (IoU + IoM) ---
    L, P = build_qubo_matrix2.qubo_matrices(boxes, scores)
    Q2 = best_alpha_2 * L - (1 - best_alpha_2) * P
    Q2 = np.round(Q2, decimals=6)
    
    t_gurobi_start = time.perf_counter()
    sol_gurobi = qubo_solver.qubo(Q2)
    duration_gurobi = time.perf_counter() - t_gurobi_start
    gurobi_times_case2.append(duration_gurobi)

    print(f"Case 2 Sol: {sol_gurobi.tolist()}")

    sol_gurobi = np.array(sol_gurobi)
    kept_indices = np.where(sol_gurobi == 1)[0]
    counts_case2.append((len(kept_indices), gt_count))

    image_predictions = []
    if len(kept_indices) > 0:
        kept_boxes = boxes[kept_indices]
        kept_scores = scores[kept_indices]
        for k in range(len(kept_boxes)):
            image_predictions.append({
                "image_id": int(image_id),
                "category_id": 1, 
                "bbox": kept_boxes[k].tolist(), 
                "score": float(kept_scores[k])
            })
    gurobi_results_case2.extend(image_predictions)

    # --- CASE 3 (IoU + Sp) ---
    L, P1, P2 = build_qubo_matrix3.qubo_matrices(boxes, scores)
    beta = (1 - best_alpha_3) / 2
    Q3 = best_alpha_3 * L - beta * P1 - beta * P2
    Q3 = np.round(Q3, decimals=6)
    
    t_gurobi_start = time.perf_counter()
    sol_gurobi = qubo_solver.qubo(Q3)
    duration_gurobi = time.perf_counter() - t_gurobi_start
    gurobi_times_case3.append(duration_gurobi)

    print(f"Case 3 Sol: {sol_gurobi.tolist()}")

    sol_gurobi = np.array(sol_gurobi)
    kept_indices = np.where(sol_gurobi == 1)[0]
    counts_case3.append((len(kept_indices), gt_count))

    image_predictions = []
    if len(kept_indices) > 0:
        kept_boxes = boxes[kept_indices]
        kept_scores = scores[kept_indices]
        for k in range(len(kept_boxes)):
            image_predictions.append({
                "image_id": int(image_id),
                "category_id": 1, 
                "bbox": kept_boxes[k].tolist(), 
                "score": float(kept_scores[k])
            })
    gurobi_results_case3.extend(image_predictions)

    # --- CASE 4 (IoU+IoM + Sp) ---
    L, P1, P2 = build_qubo_matrix4.qubo_matrices(boxes, scores)
    beta = (1 - best_alpha_4) / 2
    Q4 = best_alpha_4 * L - beta * P1 - beta * P2
    Q4 = np.round(Q4, decimals=6)
    
    t_gurobi_start = time.perf_counter()
    sol_gurobi = qubo_solver.qubo(Q4)
    duration_gurobi = time.perf_counter() - t_gurobi_start
    gurobi_times_case4.append(duration_gurobi)

    print(f"Case 4 Sol: {sol_gurobi.tolist()}")

    sol_gurobi = np.array(sol_gurobi)
    kept_indices = np.where(sol_gurobi == 1)[0]
    counts_case4.append((len(kept_indices), gt_count))

    image_predictions = []
    if len(kept_indices) > 0:
        kept_boxes = boxes[kept_indices]
        kept_scores = scores[kept_indices]
        for k in range(len(kept_boxes)):
            image_predictions.append({
                "image_id": int(image_id),
                "category_id": 1, 
                "bbox": kept_boxes[k].tolist(), 
                "score": float(kept_scores[k])
            })
    gurobi_results_case4.extend(image_predictions)

    print(f"Gurobi: Processed {i + 1} / {len(gpu_data)} images...")

# ================================================
# TABLES
# ================================================

# Prepare data
img_ids = [data['image_id'] for data in gpu_data]
box_counts = [len(data['boxes']) for data in gpu_data]

# FINAL SUMMARY TIME TABLE PER IMAGE 
table_time = PrettyTable()
table_time.title = "Gurobi - Times per Image"
table_time.field_names = ["Image ID", "Num Boxes", "Case 1 (s)", "Case 2 (s)", "Case 3 (s)", "Case 4 (s)"]

for i in range(len(img_ids)):
    table_time.add_row([
        img_ids[i], 
        box_counts[i],
        f"{gurobi_times_case1[i]:.4f}", 
        f"{gurobi_times_case2[i]:.4f}", 
        f"{gurobi_times_case3[i]:.4f}", 
        f"{gurobi_times_case4[i]:.4f}"
    ])

print("\n")
print(table_time)
print("\n")


# FINAL SUMMARY METRICS TABLE PER IMAGE 
# We structure the subsets to evaluate each image individually
subsets = [(f"ID {img_ids[i]}\n({box_counts[i]} box)", [img_ids[i]], i) for i in range(len(img_ids))]

# Lists to store scores for plotting dynamically
map_scores_c1, f1_scores_c1, mae_scores_c1 = [], [], []
map_scores_c2, f1_scores_c2, mae_scores_c2 = [], [], []
map_scores_c3, f1_scores_c3, mae_scores_c3 = [], [], []
map_scores_c4, f1_scores_c4, mae_scores_c4 = [], [], []

cases_info = [
    ('Case 1', best_alpha_1, gurobi_results_case1, counts_case1, map_scores_c1, f1_scores_c1, mae_scores_c1),
    ('Case 2', best_alpha_2, gurobi_results_case2, counts_case2, map_scores_c2, f1_scores_c2, mae_scores_c2),
    ('Case 3', best_alpha_3, gurobi_results_case3, counts_case3, map_scores_c3, f1_scores_c3, mae_scores_c3),
    ('Case 4', best_alpha_4, gurobi_results_case4, counts_case4, map_scores_c4, f1_scores_c4, mae_scores_c4)
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
table_metrics.title = "Gurobi - Metrics per Image"
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
