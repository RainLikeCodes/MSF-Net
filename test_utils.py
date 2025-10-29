import os
import torch
import torchvision.utils as vutils
import numpy as np
from models import model_utils
from utils import eval_utils

def printAngResults(accs):
    acc1=[]
    acc2=[]
    acc3=[]
    for item in accs:
        acc1.append(item['n_err_mean'])
        acc2.append(item['n_acc_15'])
        acc3.append(item['n_acc_30'])

    err_ave = sum(acc1) / len(accs)
    ave_15 = sum(acc2) / len(accs)
    ave_30 = sum(acc3) / len(accs)
    res= {
            'n_err_mean': err_ave, 
            'n_acc_15': ave_15, 
            'n_acc_30': ave_30
         } 
    print("average results:")
    print(res)

def test(args, split, loader, model, epoch):
    model.eval()
    print('---- Start %s Epoch %d: %d batches ----' % (split, epoch, len(loader)))
    accs=[]
    with torch.no_grad():
        for i, sample in enumerate(loader):
            data = model_utils.parseData(args, sample, split)
            input = model_utils.getInput(args, data)

            if split == "test":
                out_var = model(input)
            else:
                out0, out1, out_var=model(input)
                
            acc,error_map = eval_utils.calNormalAcc(data['tar'], out_var, data['m']) 
            print("err: ",acc)
            accs.append(acc)    

            # 将[-1, 1]映射到[0, 1]，即将预测值还原到正常的RGB色彩空间，因为法线图通常以颜色编码表示法线方向。
            # import torchvision.utils as vutils,它接受的张量通常是归一化到0-1范围内的RGB值
            # 许多图像处理库和一些图形API中，RGB色彩通常以8位表示，每个颜色通道的值介于0（黑色）到255（白色）之间。
            if split=='test':
                pred = (out_var.data + 1) / 2
                masked_pred = pred * data['m'].data.expand_as(out_var.data)
                GT= (data['tar']+1)/2
                save_dir='./eval/preds' 
                save_name = 'image%d.jpg' %i 
                save_name2 = 'error%d.jpg' %i 
                save_name3 = 'gt%d.jpg' %i 
                vutils.save_image(masked_pred, os.path.join(save_dir, save_name))
                vutils.save_image(error_map, os.path.join(save_dir, save_name2))
                vutils.save_image(GT, os.path.join(save_dir, save_name3))

    # 输出各指标的平均结果
    printAngResults(accs)


