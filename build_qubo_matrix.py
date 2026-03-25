import numpy as np
import IoU

def qubo_matrices(boxes, scores):
    """ 
    L and P matrices calculation to construct qubo matrix Q
    Case 1: P = IoU
    """
    num_candidates = len(scores)
    
    # matrices initialization
    P = np.zeros((num_candidates,num_candidates)) # penalty
    L = np.zeros((num_candidates,num_candidates)) # confidence score

    for i in range(num_candidates):
        # x coord horizontal top left corner
        # y coord vertical top left corner
        # w width
        # h height
        x, y, w, h = boxes[i]
        score = scores[i]
        
        L[i,i] = score
        
        # IoU calculation between pairs of boxes P(i,j) = P(j,i)
        for j in range(i+1,num_candidates):
            P[i,j] = IoU.compute_iou(boxes[i],boxes[j])
            P[j,i] = P[i,j] # symmetry

    
    return L, P
