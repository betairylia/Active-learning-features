import torch
import torch.nn
import numpy as np

import wandb

def NumEq(pred, label):
    pred_argmax = pred.detach().cpu().max(1, keepdim=True)[1]
    correct = pred_argmax.eq(label.detach().cpu().view_as(pred_argmax)).sum().item()
    return correct

def GetArgsStr(args, ignore = 'runid'):

    s = ""

    if args.keyargs == "":
        
        return None

        all_vars = list(vars(args).items())
        all_vars = sorted(all_vars, key = lambda x : x[0])

        for i in range(len(all_vars)):

            if all_vars[i][0] == ignore:
                continue

            s = s + "%s" % (all_vars[i][1])
    
    else:

        keys = args.keyargs.split(',')

        for k in keys:
            s += "%s: %s | " % (k, vars(args)[k])
        
        if len(s) > 0:
            s = s[:-3]

    s = s[:128]

    print("Argstr:\n%s" % s)
    return s

def SetMix(arr1, arr2, totalcount, proportion_arr1):

    cnt1 = int(arr1.shape[0] * proportion_arr1)
    cnt2 = totalcount - cnt1

    idx1 = torch.randperm(arr1.shape[0])[:cnt1]
    idx2 = torch.randperm(arr2.shape[0])[:cnt2]

    concated = torch.cat([arr1[idx1], arr2[idx2]], dim = 0)
    return concated[torch.randperm(totalcount)]

def Unioned(arr1, arr2):
    
    return torch.cat([arr1, arr2], dim = 0)

def ImageMosaicSQ(images, norm = True):

    from math import ceil, sqrt
    from PIL import Image

    nImages = len(images)
    imW = images[0].shape[1]
    imH = images[0].shape[2]

    nSide = ceil(sqrt(nImages))

    # Make empty big image
    elem_width = imW
    elem_height = imH
    total_width = nSide * elem_width
    total_height = nSide * elem_height

    if images[0].shape[0] == 1:
        output_im = Image.new('L', (total_width, total_height))
    else:
        output_im = Image.new('RGB', (total_width, total_height))

    if norm and isinstance(images, torch.Tensor):
        images = (images - images.min()) / (images.max() - images.min())
    else:
        iamges = images + 0.5

    # Copy generated images to big image
    for b in range(nImages):
        img = torch.clamp((images[b].detach().cpu()) * 255.0, 0.0, 255.0).permute(1, 2, 0).long().numpy()
        if images[0].shape[0] > 1:
            img = Image.fromarray(img.astype(np.uint8))
        else:
            img = Image.fromarray(img.astype(np.uint8).squeeze(-1), 'L')
        output_im.paste(img, (elem_width * (b % nSide), elem_height * (b // nSide), elem_width * (b % nSide) + elem_width, elem_height * (b // nSide) + elem_height))
        
    return output_im
