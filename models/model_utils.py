import os
import torch
import torch.nn as nn

def getInput(args, data):
    input_list = [data['input']]
    if args.in_light: input_list.append(data['l'])
    return input_list

def parseData(args, sample, split='train'):
    # input.shape: b,c*num,h,w
    input, target, mask = sample['img'], sample['N'], sample['mask'] 
    if args.cuda:
        input  = input.cuda(); target = target.cuda(); mask = mask.cuda(); 

    input_var  = torch.autograd.Variable(input)
    mask_var   = torch.autograd.Variable(mask, requires_grad=False)
    target_var = torch.autograd.Variable(target)
    
    data = {'input': input_var, 'tar': target_var, 'm': mask_var}
    
    if args.in_light:
        light = sample['light'].expand_as(input)
        if args.cuda: 
            light = light.cuda()
        light_var = torch.autograd.Variable(light)
        data['l'] = light_var
    return data 

def getInputChanel(args):
    print('[Network Input] Color image as input')
    c_in = 3
    if args.in_light:
        print('[Network Input] Adding Light direction as input')
        c_in += 3
    print('[Network Input] Input channel: {}'.format(c_in))
    return c_in

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def loadCheckpoint(path, model, cuda=True):
    # cuda
    checkpoint = torch.load(path)
    new_state_dict = {}
    model_state_dict = checkpoint["state_dict"]
    for k,v in model_state_dict.items():
        if 'module.' in k:
            name = k.replace('module.', '')
        else:
            name = k  
        new_state_dict[name]=v
    model.load_state_dict(new_state_dict)    


    
def saveCheckpoint(save_path, epoch=-1, model=None, optimizer=None, args=None):
    state   = {'state_dict': model.state_dict(), 'model': args.model}
    torch.save(state, os.path.join(save_path, 'checkp_%d.pth.tar' % (epoch)),_use_new_zipfile_serialization=False)

def conv(batchNorm, cin, cout, k=3, stride=1, pad=-1):
    pad = (k - 1) // 2 if pad < 0 else pad
    # print('Conv pad = %d' % (pad))
    if batchNorm:
        # print('=> convolutional layer with bachnorm')
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.1, inplace=True)
                )
    else:
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True),
                nn.LeakyReLU(0.1, inplace=True)
                )

def deconv(cin, cout):
    return nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
            )
