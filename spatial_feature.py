def compute_sp_feat(box1,box2):
    """This algorithm works for boxes written in (x, y, w, h) form"""
    
    # coordinates box 1
    x1_min = box1[0]
    y1_min = box1[1]
    x1_max = x1_min + box1[2] # x + w
    y1_max = y1_min + box1[3] # y + h
    area1 = box1[2] * box1[3]
    
    # coordinates box 2
    x2_min = box2[0]
    y2_min = box2[1]
    x2_max = x2_min + box2[2] # x + w
    y2_max = y2_min + box2[3] # y + h
    area2 = box2[2] * box2[3]
    
    # intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    
    intersection_area = inter_w * inter_h
    
    # sqrt of area1 * area2
    den = (area1 * area2)**0.5
    
    # do not divide by 0
    if den == 0:
        return 0
    
    return intersection_area / den
