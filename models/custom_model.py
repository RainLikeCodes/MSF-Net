from . import model_utils

def buildModel(args):
    print('Creating Model %s' % (args.model))
    in_c = model_utils.getInputChanel(args)
    other = {'img_num': args.in_img_num, 'in_light': args.in_light}
    if args.model == 'MSF_Net': 
        from models.MSF_Net import MSF_Net
        model = MSF_Net(args.fuse_type, args.use_BN, in_c, other)
    elif args.model == 'MSF_Net_run':
        from models.MSF_Net_run import MSF_Net
        model = MSF_Net(args.fuse_type, args.use_BN, in_c, other)
    else:
        raise Exception("=> Unknown Model '{}'".format(args.model))
    
    if args.cuda: 
        model = model.cuda()

    if args.retrain: # default: None 
        print("=> using pre-trained model %s" % (args.retrain))
        model_utils.loadCheckpoint(args.retrain, model, cuda=args.cuda)

    if args.resume: # default: None
        print("=> Resume loading checkpoint %s" % (args.resume))
        model_utils.loadCheckpoint(args.resume, model, cuda=args.cuda)
        
    # print(model)
    print("=> Model Parameters: %d" % (model_utils.get_n_params(model)))
    return model
