import numpy as np
import torch

def cal_scene_score(net, img, qimg, transform, network_gpu) -> float:

    img = transform(img.copy()).unsqueeze(0).cuda(network_gpu)
    qimg = transform(qimg.copy()).unsqueeze(0).cuda(network_gpu)

    with torch.no_grad():
        vecs = net(img).cpu().data.squeeze()
        qvecs = net(qimg).cpu().data.squeeze()

    vecs = vecs.numpy()
    qvecs = qvecs.numpy()

    scores = np.dot(vecs.T, qvecs)
    
    return scores


def cal_feature(net, img, transform, network_gpu) -> np.ndarray:

    with torch.no_grad():
        img = transform(img.copy()).unsqueeze(0).cuda(network_gpu)
        vecs = net(img).cpu().data.squeeze()
    
    return vecs.numpy()