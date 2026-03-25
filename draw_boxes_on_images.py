import cv2
import json
import numpy as np
import torch
from pycocotools.coco import COCO
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import RCNN

# 1. FUNCTION VISUALIZE BOXES
def visualize_boxes(image_path, boxes, solution, gt, output_filename):
    """
    Draw boxes in green and ground truth in red
    """
    image = cv2.imread(image_path)

    # colors are BGR (Blue, Green, Red), not RGB
    green = (0, 255, 0)   # green->box
    red = (0, 0, 255)     # red->gt

    for box in gt:
        x, y, w, h = [int(b) for b in box]
        cv2.rectangle(image, (x, y), (x + w, y + h), red, 2)

    for i, val in enumerate(solution):
        if val == 1:
            x, y, w, h = [int(b) for b in boxes[i]]
            cv2.rectangle(image, (x, y), (x + w, y + h), green, 2)
    # image saving
    cv2.imwrite(output_filename, image)

    print(f"IMAGE SAVED: {output_filename}")

# 2. DATA GUROBI written in form of dictionary, where you set image ID, the number of cases (1-4) and the list with sol e.g.[0,1,1,0,0,0.....]
'''sol_gurobi={Img ID: {1:[],
            2:[],
            3:[],
            4:[],
}}'''
sol_gurobi = {}

# 3. DATA SA written as Gurobi
sol_sa = {}

# 4. Faster R-CNN
instances_file = './coco2017/annotations/instances_val2017.json'
coco = COCO(instances_file)

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOADING MODEL
# COCO weights
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT  # DEFAULT means that it must use skills already developed with COCO's train 
model = fasterrcnn_resnet50_fpn(weights=weights)

# NMS parameters (let's keep all boxes for QUBO problem)
model.roi_heads.nms_thresh = 0.9  # raise the threshold to 0.9 to keep overlaps
model.roi_heads.score_thresh = 0.6 # minimum confidence score: 60%

model.to(device)
model.eval() # evaluation mode

# we print img with 42, 62 and 87 boxes
image_IDs_to_plot = [147740, 57597, 172946]

# your folder with qa solutions
json_dir = "qa_top10_results_6imm_1000reads"

# 5. IMAGES FOOR GUROBI, SA, QA
for img_id in image_IDs_to_plot:
    # img info
    img_info = coco.loadImgs(img_id)[0]
    file_name = img_info['file_name']
    image_path = f"./coco2017/val2017/{file_name}"
    
    # gt 
    annIds = coco.getAnnIds(imgIds=img_id, catIds=[1])
    anns = coco.loadAnns(annIds)
    gt_boxes = [ann['bbox'] for ann in anns]
    
    # boxes with R-CNN
    raw_boxes_rcnn, _, _ = RCNN.faster_rcnn(image_path, model, device)
    boxes_xywh = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in raw_boxes_rcnn]

    for case_num in range(1, 5):
        # Gurobi
        if img_id in sol_gurobi and case_num in sol_gurobi[img_id]:
            vec_gurobi = sol_gurobi[img_id][case_num]
            out_gurobi = f"plot_Gurobi_ID{img_id}_Case{case_num}.jpg"
            visualize_boxes(image_path, boxes_xywh, vec_gurobi, gt_boxes, out_gurobi)
            
        # SA
        if img_id in sol_sa and case_num in sol_sa[img_id]:
            vec_sa = sol_sa[img_id][case_num]
            out_sa = f"plot_SA_ID{img_id}_Case{case_num}.jpg"
            visualize_boxes(image_path, boxes_xywh, vec_sa, gt_boxes, out_sa)
        
        # QA
        json_file = f"{json_dir}/top10_case{case_num}.json"
        try:
            with open(json_file, 'r') as f:
                saved_data = json.load(f)

            # look for image in JSON
            img_data = next((item for item in saved_data if item["image_id"] == img_id), None)
            
            if img_data:
                # first vector in solutions folder
                best_vector = img_data["solutions"][0]["vector"]
                
                output_name = f"plot_QA_1000reads_ID{img_id}_Case{case_num}.jpg"
                
                # draw boxes
                visualize_boxes(image_path, boxes_xywh, best_vector, gt_boxes, output_name)
            else:
                print(f"JSON data not found for image {img_id}")
                
        except FileNotFoundError:
            print(f"File {json_file} not found.")

