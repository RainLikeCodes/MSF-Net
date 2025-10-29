import torch
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np



def calNormalAcc(gt_n, pred_n, mask=None):
    """Tensor Dim: NxCxHxW"""
    dot_product = (gt_n * pred_n).sum(1).clamp(-1,1)
    error_map   = torch.acos(dot_product) # [-pi, pi]
    angular_map = error_map * 180.0 / math.pi
    angular_map = angular_map * mask.narrow(1, 0, 1).squeeze(1)

    valid = mask.narrow(1, 0, 1).sum()
    ang_valid   = angular_map[mask.narrow(1, 0, 1).squeeze(1).bool()]
    n_err_mean  = ang_valid.sum() / valid

    n_acc_15   = (ang_valid < 15).sum().float() / valid.float()
    n_acc_30   = (ang_valid < 30).sum().float() / valid.float()

    angular_map = colorMap(angular_map.cpu().squeeze(1))
    value = {'n_err_mean': n_err_mean.item(), 
             'n_acc_15': n_acc_15.item(), 
             'n_acc_30': n_acc_30.item()
            } 

    angular_error_map = {'angular_map': angular_map} 
    return value, angular_map

def colorMap(diff, thres=90): 
    diff_norm = np.clip(diff, 0, thres) / thres
    diff_cm = torch.from_numpy(mpl.cm.jet(diff_norm.numpy()))[:,:,:, :3]
    return diff_cm.permute(0,3,1,2).clone().float()
