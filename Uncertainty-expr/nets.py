import torch
import torch.nn as nn
import torch.nn.functional as F

from recorder import Recorder

import math

class NetFactoryBase():
    
    def __init__(self, factory_args = None):
        super(NetFactoryBase, self).__init__()
        self.args = factory_args

    def getNets(self, input_shape, output_shape, args):
        raise NotImplementedError
    
class MLPFactory(NetFactoryBase):

    def __init__(self, factory_args = None):
        super(MLPFactory, self).__init__(factory_args)
        self.args = factory_args

    def initNets(self, net):

        def weights_init_wrapper(scale = 1.0):
            def weights_init(m):
                if isinstance(m, nn.Linear):
                    torch.nn.init.normal_(m.weight, mean = 0.0, std = (1 / math.sqrt(m.weight.shape[0])) * scale)
                    torch.nn.init.constant_(m.bias, 0)
            return weights_init

        net.apply(weights_init_wrapper(scale = self.args.initialization_scale))

    def getNets(self, input_shape, output_shape, hidden_dim = 1024, dropout_rate = 0.5):
        
        # Compute the input size from input_shape
        flatten_size = 1
        for dim in input_shape:
            flatten_size *= dim

        def getblock(d_in, d_out, act = torch.nn.ReLU):
            return [
                torch.nn.Linear(d_in, d_out),
                Recorder(act()) if act is not None else torch.nn.Identity(),
                torch.nn.Dropout(p = dropout_rate),
            ]

        net = torch.nn.Sequential(
            torch.nn.Flatten(),
            *getblock(flatten_size, hidden_dim),
            *getblock(hidden_dim, hidden_dim),
            *getblock(hidden_dim, hidden_dim),
        )
        
        head = torch.nn.Sequential(
            *getblock(hidden_dim, output_shape[0], act = None)
        )

        # self.initNets(net)
        # self.initNets(head)
        
        return net, head

class CNNFactory(NetFactoryBase):

    def __init__(self, factory_args = None):
        super(CNNFactory, self).__init__(factory_args)
        self.args = factory_args

    def getNets(self, input_shape, output_shape, hidden_dim = 1024, dropout_rate = 0.5):

        C = input_shape[0]
        H = input_shape[1]
        W = input_shape[2]

        p = dropout_rate

        norm_layer = torch.nn.Identity
        nd = hidden_dim // ((H//8) * (W//8))
        net = torch.nn.Sequential(
            *self.WrappedConvolutionalBlock(H, W, C, nd // 4, kernel = 5, norm = norm_layer, p = 0), # 32x32
            *self.WrappedConvolutionalBlock(H, W, nd // 4, nd // 2, kernel = 3, stride = 2, norm = norm_layer, p = p), # 16x16
            *self.WrappedConvolutionalBlock(H // 2, W // 2, nd // 2, nd // 2, kernel = 3, norm = norm_layer, p = p), # 16x16
            *self.WrappedConvolutionalBlock(H // 2, W // 2, nd // 2, nd, kernel = 3, stride = 2, norm = norm_layer, p = p), # 8x8
            *self.WrappedConvolutionalBlock(H // 4, W // 4, nd, nd, kernel = 3, norm = norm_layer, p = p), # 8x8
            *self.WrappedConvolutionalBlock(H // 4, W // 4, nd, nd, kernel = 3, stride = 2, norm = norm_layer, p = p), # 4x4
            *self.WrappedConvolutionalBlock(H // 8, W // 8, nd, nd, kernel = 3, norm = norm_layer, p = p), # 4x4
            torch.nn.Flatten(),
        )
        
        head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, output_shape[0])
        )
        
        return net, head

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def WrappedConvolutionalBlock(self, h, w, in_ch, out_ch, norm = torch.nn.InstanceNorm2d,\
                                  kernel = 3, stride = 1, act = True, p = 0.5):
        
        conv = nn.Conv2d(in_ch, out_ch, kernel, stride = stride, 
                         padding = math.ceil(self.calc_same_pad(64, kernel, stride, 1) / 2))
        
        if norm is torch.nn.LayerNorm:
            bn = norm([out_ch, h // stride, w // stride])
        else:
            bn = norm(out_ch)
        
        if act:
            return [conv, bn, Recorder(nn.ReLU(inplace = True)), nn.Dropout2d(p = p)]
        else:
            return [conv, bn, nn.Dropout2d(p = p)]

net_dict =\
{
    'mlp': MLPFactory,
    'simple-cnn': CNNFactory,
}
