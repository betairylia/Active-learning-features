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

    def register_model(self, model):
        pass

    def prediction_reset(self, model):
        pass

    # def custom_on_prediction_start(self, model):
    #     pass

    # def custom_prediction_step(self, model, batch, batch_idx):
    #     pass

    # def custom_query_step(self, budget, evidences, net):
    #     pass

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

            batch = batch.detach().to(net.head[0].weight.device)

            # hist, bin_edges = torch.histogram()
            nonzeroCount = - (batch > 0).sum(dim = (1,2))
            result.append(nonzeroCount.detach().cpu())

        uncertainties = torch.cat(result, dim = 0)
        # print(result.shape)
        indices = np.argsort(uncertainties.numpy())
        indices = indices[-budget:]

        # centers, indices = kmeans_plusplus(result.numpy(), n_clusters = budget, random_state = 0)
        return indices

class MonteCarloBound(AdvancedAbstractHeuristic):

    def register_model(self, model):
        
        self.model = model

        self.net_layer_ids = []
        self.head_layer_ids = []

        self.net_sums = []
        self.head_sums = []

        for i, layer in enumerate(model.net):
            if hasattr(layer, "weight"):
                self.net_layer_ids.append(i)
                self.net_sums.append(torch.zeros_like(layer.weight))

        for i, layer in enumerate(model.head):
            if hasattr(layer, "weight"):
                self.head_layer_ids.append(i)
                self.head_sums.append(torch.zeros_like(layer.weight))

    def prediction_reset(self):

        print("Prediction reset")

        for i in range(len(self.net_sums)):
            self.net_sums[i].fill_(0)
        
        for i in range(len(self.head_sums)):
            self.head_sums[i].fill_(0)

    def custom_prediction_step(self, model, batch, batch_idx):

        # for bid in range(batch.shape[0]):

        model.zero_grad()

        # Forward
        x, _ = batch
        feat = model.net_no_dropout(x)
        # logits = model.head(feat)

        # Activation
        feat_aligned = feat.unsqueeze(-1) # [bs, hidden_dim, 1]
        last_layer = model.head[0]
        activation = feat_aligned * last_layer.weight.permute(1, 0).unsqueeze(0) + last_layer.bias[None, None, :] # [bs, hidden_dim, out_dim]

        activation_dev = activation - activation.mean(1, keepdims = True)
        loss_proxy = (activation * activation_dev.detach()).mean((1, 2)).sum()
        loss_proxy.backward()

        # Gradient
        for i, li in enumerate(self.net_layer_ids):
            grad = model.net[li].grad # [hidden_dim, input_dim]
            self.net_sums[i] += grad

        for i, li in enumerate(self.head_layer_ids):
            grad = model.head[li].grad
            self.head_sums[i] += grad # TODO: Sum to 1 class?

        return None

    def custom_query_step(self, budget, evidences, dataloader, model):
        
        scores = []

        for batch in dataloader:

            xs, _ = batch
    
            for bid in range(xs.shape[0]):
    
                model.zero_grad()
                x = xs[bid].unsqueeze(0)

                feat = model.net_no_dropout(x)
                out = model.head(feat)
                fake_label = torch.argmax(out, dim = -1)

                loss = model.loss(x, fake_label)
                loss.backward()

                score = 0
                # Gradient
                for i, li in enumerate(self.net_layer_ids):
                    grad = model.net[li].grad # [hidden_dim, input_dim]
                    score += (self.net_sums[i] * grad).sum().detach().cpu()

                for i, li in enumerate(self.head_layer_ids):
                    grad = model.head[li].grad
                    score += (self.head_sums[i] * grad).sum().detach().cpu()

                scores.append(score)
        
        scores = np.asarray(scores)
        indices = np.argsort(scores)
        indices = indices[-budget:] # Largest scores being picked

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
        "mcgradient": MonteCarloBound,
    }[name](shuffle_prop=shuffle_prop, reduction=reduction, **kwargs)
    return heuristic
