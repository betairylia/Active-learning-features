import torch
from torch import nn
from torch.nn import functional as F

from functools import partial
from functorch import make_functional_with_buffers, vmap, vjp, jvp, jacrev
from .paramsutils import *

import math
import copy

import utils
from . import filters

class NTKHelper():

    def __init__(
        self, 
        net, 
        outdim = -1, 
        dirc = 'rev',
        filtering = filters.weight_only_ignore_bn   # Func: (name: str, shape: List[Number]) -> Boolean
                                                    # will only use parameters that passes the filtering function.
                                                    # can be used to filter-out unwanted layers.
    ):

        self.refresh(net)

        self.outdim = outdim
        self.filter = filtering

        if dirc == 'rev':
            self.jac = jacrev
        elif dirc == 'fwd':
            self.jac = jacfwd
        else:
            utils.log("NTKHelper: dirc can only be 'rev' or 'fwd'!")
            raise
    
    def refresh(self, net):
        self.fnet, self.fparams, self.fbuffer = make_functional_with_buffers(net)
        self.fparams_names = dict(net.named_parameters()).keys()
        self.fbuffers_names = dict(net.named_buffers()).keys()
        self.fnet_single = self.get_fnet_single()
 
    # batchsize 1
    def get_fnet_single(self):

        def foo(params, x):
            result = self.fnet(params, self.fbuffer, x.unsqueeze(0)).squeeze(0)
            if self.outdim < 0:
                return result
            else:
                return result[self.outdim:self.outdim+1]
        
        return foo
    
    # Filter any tuples that corresponds to fparams
    # e.g., computed Jacobian
    def filter_param_tuples(self, t):
        if(len(t) == len(self.fparams_names)):
            t = [x for n, x in zip(self.fparams_names, t) if self.filter(n, x.shape)]
            return t
        else:
            return t
    
    def proxy_loader(self, x, bs):
        batches = len(x) // bs
        for i in range(batches):
            si = i * bs
            ei = si + bs
            yield x[si:ei]

    '''
    x1 - Tensor[Nx1, *d_in] or dataloader (will ignore NTK_batchsize)
    x2 - Tensor[Nx2, *d_in] or dataloader (will ignore NTK_batchsize)
    
    returns:
    - 'pairwise':   Tensor[Nx1, Nx2, *NTK_shape]    (Compute NTK for each pair x1[i], x2[j] for all i, j)
    - '1to1':       Tensor[Nx1, *NTK_shape]         (Compute NTK only for x1[i], x2[i] for all i, requires Nx1 == Nx2)

    NTK_shape:
    - 'full':       [d_out, d_out]
    - 'diagonal':   [d_out]
    - 'trace':      Scalar (squeezed)
    '''
    def compute_ntk(
        self, x1, x2, 
        mode = 'trace', batch_mode = 'pairwise', 
        NTK_batchsize = 4,
        x1_map = lambda x : x,
        x2_map = lambda x : x):

        # TODO: Support partial batches
        # assert len(x1) % NTK_batchsize == 0 and len(x2) % NTK_batchsize == 0

        x1dl = x1
        x2dl = x2

        if isinstance(x1, torch.Tensor):
            x1dl = self.proxy_loader(x1, NTK_batchsize)
        
        if isinstance(x2, torch.Tensor):
            x2dl = self.proxy_loader(x2, NTK_batchsize)

        # Fill arrays

        if batch_mode == 'pairwise':

            result = []
            for bx1 in x1dl:
                row_result = []
                for bx2 in x2dl:
                    row_result.append(
                        self.compute_ntk_eval_batch(
                            x1_map(bx1),
                            x2_map(bx2),
                            mode,
                            batch_mode
                        )
                    )
                row_result = torch.cat(row_result, dim = 1)
                result.append(row_result)
            result = torch.cat(result, dim = 0)
            
            return result

        elif batch_mode == '1to1':

            result = []
            for bx1, bx2 in zip(x1dl, x2dl):
                result.append(
                    self.compute_ntk_eval_batch(
                        x1_map(bx1),
                        x2_map(bx2),
                        mode,
                        batch_mode
                    )
                )
            result = torch.cat(result)
            
            return result

    # TODO: Filter buffers?
    # Same as above function but don't split to smaller batches
    def compute_ntk_eval_batch(self, x1, x2, mode = 'trace', batch_mode = 'pairwise'):

        x1 = x1.to(self.fparams[0].device)
        x2 = x2.to(self.fparams[0].device)

        # Jacobian for x1
        jac1 =\
            vmap(                               # Maps following operations to support batches
                self.jac(self.fnet_single),     # Computes the Jacobian of outputs w.r.t. first arguments ('params')  
                (None, 0)                       # Batches grows None (no batches) for first arg ('params'), grows on dim 0 for second arg ('x')
            )(self.fparams, x1)                 # Feed inputs: parameters and x
                                                # Output: tuples of Tensor[batch_size(x1), output_dim, *params.shape]
                                                # e.g., for a conv weight with [64, 64, 3, 3] shape, 10 classification, x1 with 2 datums:
                                                #          => output is [2, 10, 64, 64, 3, 3]
        jac1 = self.filter_param_tuples(jac1)
        jac1 = [j.flatten(2) for j in jac1]     # Converts to [bs, dim_o, Nparams]

        # Jacobian for x2
        jac2 = vmap(self.jac(self.fnet_single), (None, 0))(self.fparams, x2)
        jac2 = self.filter_param_tuples(jac2)
        jac2 = [j.flatten(2) for j in jac2]

        # Compute J(x1) @ J(x2).T

        einsum_expr_param = ''
        einsum_expr_lhs_param = ('a', 'a')

        if mode == 'full':
            einsum_expr_param = 'ab'
            einsum_expr_lhs_param = ('a', 'b')
        elif mode == 'trace':
            einsum_expr_param = ''
        elif mode == 'diagonal':
            einsum_expr_param = 'a'
        else:
            assert False
        
        einsum_expr_batch = 'NM'
        einsum_expr_lhs_batch = ('N', 'M')

        if batch_mode == 'pairwise':
            pass
        elif batch_mode == '1to1':
            einsum_expr_batch = 'N'
            einsum_expr_lhs_batch = ('N', 'N')
        else:
            assert False
        
        einsum_expr = '%s%sf,%s%sf->%s%s' % (
            einsum_expr_lhs_batch[0],
            einsum_expr_lhs_param[0],
            einsum_expr_lhs_batch[1],
            einsum_expr_lhs_param[1],
            einsum_expr_batch,
            einsum_expr_param
        )

        result = torch.stack([torch.einsum(einsum_expr, j1, j2) for j1, j2 in zip(jac1, jac2)])
        result = result.sum(0) # TODO: Layer-wise scaling?

        return result.detach()
