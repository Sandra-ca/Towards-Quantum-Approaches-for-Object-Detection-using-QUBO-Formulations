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

# SOLUTION TO KERNEL CRASH: ignore conflicts between OpenMP libraries (Gurobi vs. PyTorch/NumPy)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import build_qubo_matrix 
import build_qubo_matrix2
import build_qubo_matrix3
import build_qubo_matrix4
import metrics
import RCNN
import logging
import brute_force

# silenced torchvision/pytorch
logging.getLogger("torchvision").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# path to the file containing the ground truths for each image (called with an ID)
instances_file = './coco2017/annotations/instances_val2017.json'
coco = COCO(instances_file) #initialization

# select only images that have people (and that have people as ground truth)
catIds = coco.getCatIds(catNms=['person'])
image_IDs = coco.getImgIds(catIds=catIds) # Image IDs

# ========================================================
# image ID we like to analyze
# !!! use the instance 213035 (31 box) only with GPU
image_IDs = [532481, 270908, 458755, 213035] # 5, 14, 23, 13 boxes
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
    
    valid_count +=1

    boxes_xywh = []
    for box in raw_boxes:
        x1, y1, x2, y2 = box
        w = x2 - x1  
        h = y2 - y1  
        boxes_xywh.append([x1, y1, w, h]) 
    
    print(f"[{valid_count}] Id imm: {img_id} | Nome imm: {file_name} | Box trovate: {len(raw_boxes)} | Tempo RCNN: {t_rcnn_end - t_rcnn_start:.3f}s")

    gpu_data.append({
        'image_id': img_id,
        'boxes': np.array(boxes_xywh),
        'scores': scores,
        'file_name': file_name
    })


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

# best alpha computed using best_alpha_gurobi.py
best_alpha_1 = 0.58
best_alpha_2 = 0.60
best_alpha_3 = 0.60
best_alpha_4 = 0.62

# lists for times
brute_times_case1 = []
brute_times_case2 = []
brute_times_case3 = []
brute_times_case4 = []

# lists for predictions
brute_results_case1 = []
brute_results_case2 = []
brute_results_case3 = []
brute_results_case4 = []

# lists for pred_count and gt_count for MAE
counts_case1 = []
counts_case2 = []
counts_case3 = []
counts_case4 = []

# WARM-UP
dummy_Q = np.random.rand(3, 3).astype(np.float64)

if device.type == 'cuda':
    _, _ = brute_force.qubo_brute_gpu(dummy_Q)
    torch.cuda.synchronize()
else:
    _, _ = brute_force.qubo_brute(dummy_Q)


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

    print(f"\n--- Image ID {image_id} ({N} box) ---")

    # --- CASE 1 (IoU) ---
    L, P = build_qubo_matrix.qubo_matrices(boxes, scores)
    Q1 = best_alpha_1 * L - (1 - best_alpha_1) * P
    Q1 = np.round(Q1, decimals=6)

    t_start = time.perf_counter()
    if device.type == 'cuda':
        sol, val = brute_force.qubo_brute_gpu(Q1)
        torch.cuda.synchronize()
    else:
        sol, val = brute_force.qubo_brute(Q1)
    brute_times_case1.append(time.perf_counter() - t_start)
    
    sol = np.array(sol)

    # print sol and energy(with -) case 1
    print(f"Case 1 Sol: {sol.tolist()} with energy: {-val:.6f}")

    # collect indices of the kept boxes
    kept_indices = np.where(sol == 1)[0]
    counts_case1.append((len(kept_indices), gt_count)) # for MAE
        
    # I save the boxes, scores and labels corresponding to the indexes, if I don't have any boxes we skip this step
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
    brute_results_case1.extend(image_predictions)


    # --- CASE 2 (IoU + IoM) ---
    L, P = build_qubo_matrix2.qubo_matrices(boxes, scores)
    Q2 = best_alpha_2 * L - (1 - best_alpha_2) * P
    Q2 = np.round(Q2, decimals=6)

    t_start = time.perf_counter()
    if device.type == 'cuda':
        sol, val = brute_force.qubo_brute_gpu(Q2)
        torch.cuda.synchronize()
    else:
        sol, val = brute_force.qubo_brute(Q2)
    brute_times_case2.append(time.perf_counter() - t_start)
    
    sol = np.array(sol)

    # print sol and energy(with -) case 2
    print(f"Case 2 Sol: {sol.tolist()} with energy: {-val:.6f}")

    # collect indices of the kept boxes
    kept_indices = np.where(sol == 1)[0]
    counts_case2.append((len(kept_indices), gt_count)) # for MAE

    # I save the boxes, scores and labels corresponding to the indexes, if I don't have any boxes we skip this step
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
    brute_results_case2.extend(image_predictions)


    # --- CASE 3 (IoU + Sp) ---
    L, P1, P2 = build_qubo_matrix3.qubo_matrices(boxes, scores)
    beta = (1 - best_alpha_3) / 2
    Q3 = best_alpha_3 * L - beta * P1 - beta * P2
    Q3 = np.round(Q3, decimals=6)

    t_start = time.perf_counter()
    if device.type == 'cuda':
        sol, val = brute_force.qubo_brute_gpu(Q3)
        torch.cuda.synchronize()
    else:
        sol, val = brute_force.qubo_brute(Q3)
    brute_times_case3.append(time.perf_counter() - t_start)
    
    sol = np.array(sol)

    # print sol and energy(with -) case 3
    print(f"Case 3 Sol: {sol.tolist()} with energy: {-val:.6f}")

    # collect indices of the kept boxes
    kept_indices = np.where(sol == 1)[0]
    counts_case3.append((len(kept_indices), gt_count)) # for MAE
        
    # I save the boxes, scores and labels corresponding to the indexes, if I don't have any boxes we skip this step
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
    brute_results_case3.extend(image_predictions)


    # --- CASE 4 (IoU+IoM + Sp) ---
    L, P1, P2 = build_qubo_matrix4.qubo_matrices(boxes, scores)
    beta = (1 - best_alpha_4) / 2
    Q4 = best_alpha_4 * L - beta * P1 - beta * P2
    Q4 = np.round(Q4, decimals=6)

    t_start = time.perf_counter()
    if device.type == 'cuda':
        sol, val = brute_force.qubo_brute_gpu(Q4)
        torch.cuda.synchronize()
    else:
        sol, val = brute_force.qubo_brute(Q4)
    brute_times_case4.append(time.perf_counter() - t_start)
    
    sol = np.array(sol)

    # print sol and energy(with -) case 4
    print(f"Case 4 Sol: {sol.tolist()} with energy: {-val:.6f}")

    # collect indices of the kept boxes
    kept_indices = np.where(sol == 1)[0]
    counts_case4.append((len(kept_indices), gt_count)) # for MAE
    
    # I save the boxes, scores and labels corresponding to the indexes, if I don't have any boxes we skip this step
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
    brute_results_case4.extend(image_predictions)

    print(f"Brute force: Processed {i + 1} / {len(gpu_data)} images...")



# --- FINAL SUMMARY TIME TABLE ---
table_brute = PrettyTable()
table_brute.title = "Brute Force - Execution Times"

n_boxes_list = [len(data['boxes']) for data in gpu_data]
header_times = ["Config."] + [f"{n}-box (s)" for n in n_boxes_list]
table_brute.field_names = header_times

table_brute.add_row(["Case 1"] + [f"{t:.4f}" for t in brute_times_case1])
table_brute.add_row(["Case 2"] + [f"{t:.4f}" for t in brute_times_case2])
table_brute.add_row(["Case 3"] + [f"{t:.4f}" for t in brute_times_case3])
table_brute.add_row(["Case 4"] + [f"{t:.4f}" for t in brute_times_case4])

print(table_brute)
print("\n")


# --- FINAL SUMMARY METRICS TABLE ---
table_metrics = PrettyTable()
table_metrics.title = "Brute Force - Metrics"
table_metrics.field_names = ["n. boxes", "Config.", "F1", "mAP(std)", "mAP(.50)", "mAR(10)", "mAR(100)", "MAE"]

cases_info = [
    ('Case 1', brute_results_case1, counts_case1),
    ('Case 2', brute_results_case2, counts_case2),
    ('Case 3', brute_results_case3, counts_case3),
    ('Case 4', brute_results_case4, counts_case4)
]

original_stdout = sys.stdout

for i, data in enumerate(gpu_data):
    n_boxes = len(data['boxes'])
    img_id = data['image_id']

    for case_idx, (case_name, preds_all, counts_all) in enumerate(cases_info):
        
        preds_img = [p for p in preds_all if p['image_id'] == img_id]
        
        kept_boxes, gt_boxes = counts_all[i]
        mae = abs(kept_boxes - gt_boxes)
        
        if len(preds_img) == 0:
            f1, mAP_std, mAP_50, mAR_10, mAR_100 = 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            coco_dt = coco.loadRes(preds_img)
            coco_eval = COCOeval(coco, coco_dt, 'bbox')
            coco_eval.params.imgIds = [img_id]
            coco_eval.params.catIds = [1]
            
            sys.stdout = open(os.devnull, 'w') # silence print
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
                
            precision, recall, f1 = metrics.compute_metrics(coco, preds_img, [img_id])

        col_n_boxes = str(n_boxes) if case_idx == 0 else ""

        table_metrics.add_row([
            col_n_boxes,
            case_name,
            f"{f1:.4f}",
            f"{mAP_std:.4f}",
            f"{mAP_50:.4f}",
            f"{mAR_10:.4f}",
            f"{mAR_100:.4f}",
            f"{mae:.3f}"
        ])
        
    if i < len(gpu_data) - 1:
        table_metrics.add_row(["-"*8, "-"*8, "-"*6, "-"*8, "-"*8, "-"*8, "-"*8, "-"*6])

print(table_metrics)
print("\n")