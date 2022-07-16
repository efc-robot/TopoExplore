import numpy as np


def predict_navigable_depth(depth, dim) -> np.ndarray:
    shape = depth.shape
    step = int(shape[0] / dim)
    hist = np.ndarray(dim)
    for i in range(dim):
        hist[i] = np.sum(depth[:, i*step:(i+1)*step-1])
    hist = (-1) * hist
    return hist / np.sum(hist)
    #return hist
    

def  predict_navigable_3d(threeD, dim) -> np.ndarray:
    depth = np.asarray(threeD[0][ :, :, 2])
    return predict_navigable_depth(depth, dim)

def predict_navigable_panorama(Depth, dim) -> list:
    return [predict_navigable_depth(depth, dim) for depth in Depth]
