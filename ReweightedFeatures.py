import torch
import numpy as np

def ReweightFeatures(features, weights, relu = False):
    
    '''
    Reweights features accroading to weight.
    features: [N_batches (Nb), dim]
    weight:   [dim,]
    '''
    
    # Apply weight sign and sort features and abs weights
    features_signed = features * (torch.sign(weights))
    features_sorted, features_order = torch.sort(features_signed)
    weights_sorted = torch.abs(weights)[features_order] # => [Nb, dim]
    
    if relu:
        weights_sorted[features_sorted == 0] = 0
    
    # Collect prefix-sum of weight vectors, with an appended zero in front
    weights_cumsum = torch.cumsum(torch.abs(weights_sorted), dim = 1)
    weights_cumsum = torch.cat(
        [
            torch.zeros((weights_cumsum.shape[0], 1), device = weights_cumsum.device), 
            weights_cumsum
        ],
        dim = 1
    )
    
    # Create uniform-sampled points for reweighting
    weights_total = weights_cumsum[:, -1]
    uniformed = torch.linspace(start = 0, end = weights_total[0], device = weights.device, steps = weights.shape[0])#.unsqueeze(0) * weights_total.unsqueeze(1)
    uniformed = uniformed.unsqueeze(0).repeat(weights_cumsum.shape[0], 1)

    # Perform binary search to find interpolation ends
    searched_results = torch.searchsorted(weights_cumsum, uniformed)
    searched_results[:, 0] = 1 # Remove first 0's 
    
    # Linear interpolation: starts[ <------------ interp --> ] ends
    starts = torch.gather(features_sorted, -1, searched_results - 1)
    ends = torch.gather(features_sorted, -1, torch.minimum(searched_results, torch.ones((1,), dtype = torch.long, device = features_sorted.device) * (features_sorted.shape[-1] - 1)))

    # Linear interpolation: obtain interp from both weight ends
    weights_s = torch.gather(weights_cumsum, -1, searched_results - 1)
    weights_e = torch.gather(weights_cumsum, -1, searched_results)
    interp = (uniformed - weights_s) / (weights_e - weights_s)
    
    # Do the interpolation
    result = starts + (ends - starts) * interp
    return result