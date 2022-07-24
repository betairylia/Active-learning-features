from math import ceil
from typing import Union, Callable

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

import baal.active.heuristics as heuristics

class AdvancedAbstractHeuristic(heuristics.AbstractHeuristic):
    
    def compute_score(self, predictions, **kwargs):
        """
        Compute the score according to the heuristic.
        Args:
            predictions (ndarray): Array of predictions
        Returns:
            Array of scores.
        """
        return super().compute_score(predictions)

    def get_uncertainties_generator(self, predictions, **kwargs):
        """
        Compute the score according to the heuristic.
        Args:
            predictions (Iterable): Generator of predictions
        Raises:
            ValueError if the generator is empty.
        Returns:
            Array of scores.
        """
        return super().get_uncertainties_generator(predictions)

    def get_uncertainties(self, predictions, **kwargs):
        """
        Get the uncertainties.
        Args:
            predictions (ndarray): Array of predictions
        Returns:
            Array of uncertainties
        """
        return super().get_uncertainties(predictions)

    def get_indices(self, budget, predictions, features, net, **kwargs):
        pass

    def reorder_indices(self, scores, **kwargs):
        """
        Order indices given their uncertainty score.
        Args:
            scores (ndarray/ List[ndarray]): Array of uncertainties or
                list of arrays.
        Returns:
            ordered index according to the uncertainty (highest to lowes).
        Raises:
            ValueError if `scores` is not uni-dimensional.
        """
        return super().reorder_indices(predictions)

    def get_ranks(self, predictions, **kwargs):
        """
        Rank the predictions according to their uncertainties.
        Args:
            predictions (ndarray): [batch_size, C, ..., Iterations]
        Returns:
            Ranked index according to the uncertainty (highest to lowes).
            Scores for all predictions.
        """
        return super().get_ranks(predictions)

    def __call__(self, predictions, **kwargs):
        """Rank the predictions according to their uncertainties.
        Only return the scores and not the associated uncertainties.
        """
        return self.get_ranks(predictions, **kwargs)[0]

class BADGE(AdvancedAbstractHeuristic):
    # TODO
    pass

class BAIT(AdvancedAbstractHeuristic):
    # TODO
    pass

from sklearn.cluster import kmeans_plusplus

class FeatureDistTest(AdvancedAbstractHeuristic):

    def get_indices(self, budget, predictions, features, net, **kwargs):

        features = features.permute((0, 2, 1))

        N = features.shape[0]

        batchsize = 256
        nBatch = ceil(N / batchsize)

        # Sort
        result = []
        for bi in tqdm(range(nBatch)):
            if batchsize*(bi+1) >= N:
                batch = features[batchsize*bi:]
            else:
                batch = features[batchsize*bi:batchsize*(bi+1)]

            batch = batch.detach().to(net[0].weight.device)

            sortedbatch, idx = torch.sort(batch, dim = -1)
            result.append(sortedbatch.mean(dim = 1).detach().cpu())

        result = torch.cat(result, dim = 0)
        # print(result.shape)
        
        centers, indices = kmeans_plusplus(result.numpy(), n_clusters = budget, random_state = 0)
        return indices

class FeatureDistNonzero(AdvancedAbstractHeuristic):

    def get_indices(self, budget, predictions, features, net, **kwargs):

        features = features.permute((0, 2, 1))

        N = features.shape[0]

        batchsize = 256
        nBatch = ceil(N / batchsize)

        # Sort
        result = []
        for bi in tqdm(range(nBatch)):
            if batchsize*(bi+1) >= N:
                batch = features[batchsize*bi:]
            else:
                batch = features[batchsize*bi:batchsize*(bi+1)]

            batch = batch.detach().to(net[0].weight.device)

            # hist, bin_edges = torch.histogram()
            nonzeroCount = (batch > 0).sum(dim = (1,2))
            result.append(nonzeroCount.detach().cpu())

        uncertainties = torch.cat(result, dim = 0)
        # print(result.shape)
        indices = np.argsort(uncertainties.numpy())
        indices = indices[-budget:]

        # centers, indices = kmeans_plusplus(result.numpy(), n_clusters = budget, random_state = 0)
        return indices


class ReweightedFeatureDistTest(AdvancedAbstractHeuristic):
    # TODO
    pass

def get_heuristic_with_advanced(
    name: str, shuffle_prop: float = 0.0, reduction: Union[str, Callable] = "none", **kwargs
) -> heuristics.AbstractHeuristic:
    """
    Create an heuristic object from the name.

    Args:
        name (str): Name of the heuristic.
        shuffle_prop (float): Shuffling proportion when getting ranks.
        reduction (Union[str, Callable]): Reduction used after computing the score.
        kwargs (dict): Complementary arguments.

    Returns:
        AbstractHeuristic object.
    """
    heuristic: heuristics.AbstractHeuristic = {
        "random": heuristics.Random,
        "certainty": heuristics.Certainty,
        "entropy": heuristics.Entropy,
        "margin": heuristics.Margin,
        "bald": heuristics.BALD,
        "variance": heuristics.Variance,
        "precomputed": heuristics.Precomputed,
        "batch_bald": heuristics.BatchBALD,
        "fdist": FeatureDistTest,
        "fdist-nonzero": FeatureDistNonzero,
    }[name](shuffle_prop=shuffle_prop, reduction=reduction, **kwargs)
    return heuristic
