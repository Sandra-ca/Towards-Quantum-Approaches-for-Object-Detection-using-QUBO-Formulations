import numpy as np
import IoU
import IoM
import spatial_feature

def qubo_matrices(boxes, scores):
    """ 
    L, P1 and P2 matrices calculation to construct qubo matrix Q
    P1 = 0.7*IoU + 0.3*IoM, P2 = Sp. Feat.
    """
    num_candidates = len(scores)

    # matrices initialization
    P1 = np.zeros((num_candidates,num_candidates)) # penalty
    P2 = np.zeros((num_candidates,num_candidates)) # penalty
    L = np.zeros((num_candidates,num_candidates)) # confidence score

    for i in range(num_candidates):
        # x coord horizontal top left corner
        # y coord vertical top left corner
        # w width
        # h height
        x, y, w, h = boxes[i] 
        score = scores[i]
        
        L[i,i] = score
        
        # 0.7*IoU + 0.3*IoM calculation between pairs of boxes P1(i,j) = P1(j,i)
        for j in range(i+1,num_candidates):
            P1[i,j] = 0.7*IoU.compute_iou(boxes[i],boxes[j]) + 0.3*IoM.compute_iom(boxes[i],boxes[j])
            P1[j,i] = P1[i,j] # symmetry

        # spatial feature calculation between pairs of boxes P2(i,j) e P2(j,i)
        for j in range(i+1,num_candidates):
            P2[i,j] = spatial_feature.compute_sp_feat(boxes[i],boxes[j])
            P2[j,i] = P2[i,j] # symmetry
    
    return L, P1, P2 