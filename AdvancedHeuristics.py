from math import ceil
from typing import Union, Callable

import numpy as np
import torch
from torch import Tensor
from torch import nn
from tqdm import tqdm

import random

import baal.active.heuristics as heuristics

from opacus.grad_sample import GradSampleModule

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

#         self.net_layer_ids = []
#         self.head_layer_ids = []

#         self.net_sums = []
#         self.head_sums = []

        self.key_layers = model.key_layers
        self.sums = []

#         for i, layer in enumerate(model.net):
#             if hasattr(layer, "parameters") and len(list(layer.parameters())) > 0:
#                 self.net_layer_ids.append(i)
#                 self.net_sums.append(torch.zeros_like(next(layer.parameters())))

#         for i, layer in enumerate(model.head):
#             if hasattr(layer, "parameters") and len(list(layer.parameters())) > 0:
#                 self.head_layer_ids.append(i)
#                 self.head_sums.append(torch.zeros_like(next(layer.parameters())))
        
        for layer in self.key_layers:
            
            for param in layer.parameters():
                if len(param.shape) > 1:
                    break
            else:
                continue
            
            print("Registered: [%s]" % (str(param.shape)))
            self.sums.append(torch.zeros_like(param))

    def prediction_reset(self):

        print("Prediction reset")

#         for i in range(len(self.net_sums)):
#             self.net_sums[i].fill_(0)
        
#         for i in range(len(self.head_sums)):
#             self.head_sums[i].fill_(0)
        
        for i in range(len(self.sums)):
            self.sums[i].fill_(0)

    def custom_prediction_step(self, model, batch, batch_idx):

        # for bid in range(batch.shape[0]):
#         for gs in model.gradsamples:
# #             gs.disable_hooks()
#             pass
        
        model.zero_grad()
        model.evalUncertain()

        # Forward
        x, _ = batch
        feat = model.net_no_dropout(x)
        logits = model.head(feat)

        # Activation
        feat_aligned = feat.unsqueeze(-1) # [bs, hidden_dim, 1]
        last_layer = model.head[-1]
        activation = feat_aligned * last_layer.weight.permute(1, 0).unsqueeze(0) + last_layer.bias[None, None, :] # [bs, hidden_dim, out_dim]

        activation_dev = activation - activation.mean(1, keepdims = True)
        loss_proxy = (activation * activation_dev.detach()).mean((1, 2)).sum()
        loss_proxy.backward()

        # Gradient
        for i, layer in enumerate(self.key_layers):
            
            for param in layer.parameters():
                if len(param.shape) > 1:
                    break
            else:
                continue
            
            grad = param.grad # [hidden_dim, input_dim]
            self.sums[i] = self.sums[i].to(grad.device)
            self.sums[i] += grad
            
#         for i, li in enumerate(self.net_layer_ids):
#             grad = next(model.net[li].parameters()).grad # [hidden_dim, input_dim]
#             self.net_sums[i] = self.net_sums[i].to(grad.device)
#             self.net_sums[i] += grad
            
#         for i, li in enumerate(self.head_layer_ids):
#             grad = next(model.head[li].parameters()).grad
#             self.head_sums[i] = self.head_sums[i].to(grad.device)
#             self.head_sums[i] += grad # TODO: Sum to 1 class?

#         for gs in model.gradsamples:
# #             gs.enable_hooks()
#             pass
        
        model.unevalUncertain()

        return None

    def custom_query_step(self, budget, evidences, dataloader, model):
        
        scores = []

        for batch in tqdm(dataloader, "Collecting query scores"):

            xs, _ = batch
            xs = xs.to(next(model.head[self.head_layer_ids[0]].parameters()).device)

            for bid in range(xs.shape[0]):

                model.zero_grad()
                x = xs[bid].unsqueeze(0)

                feat = model.net_no_dropout(x)
                out = model.head(feat)
                fake_label = torch.argmax(out, dim = -1)

                loss = model.loss(out, fake_label)
                loss.backward()

                score = 0
                # Gradient
                for i, layer in enumerate(self.key_layers):
                    
                    for param in layer.parameters():
                        if len(param.shape) > 1:
                            break
                    else:
                        continue
                    
                    grad = param.grad # [hidden_dim, input_dim]
                    self.sums[i] = self.sums[i].to(grad.device)
                    score += (self.sums[i] * grad).sum().detach().cpu()
#                 for i, li in enumerate(self.net_layer_ids):
#                     grad = next(model.net[li].parameters()).grad # [hidden_dim, input_dim]
#                     self.net_sums[i] = self.net_sums[i].to(grad.device)
#                     score += (self.net_sums[i] * grad).sum().detach().cpu()

#                 for i, li in enumerate(self.head_layer_ids):
#                     grad = next(model.head[li].parameters()).grad
#                     self.head_sums[i] = self.head_sums[i].to(grad.device)
#                     score += (self.head_sums[i] * grad).sum().detach().cpu()

                scores.append(-score)

        scores = np.asarray(scores)
        indices = np.argsort(scores)
        indices = indices[-budget:] # Largest scores being picked

        return indices

    
    
    
    
class MonteCarloBoundBatched(MonteCarloBound):
    
    def get_data_vec(self, model, net, head, keys, x):
        
        x = x.unsqueeze(0)
        return self.get_batch_vec(model, net, head, keys, x).flatten()
    
    def get_batch_vec(self, model, net, head, keys, xs):
        
        # net / head: GradSampleModule
        net.zero_grad()
        head.zero_grad()
        
        # net / head: nn.Sequential
        net = next(net.children())
        head = next(head.children())
        
        bs = xs.shape[0]

        feat = net(xs)
        out = head(feat)
        fake_label = torch.argmax(out, dim = -1)

        loss = model.loss(out, fake_label)
        loss.backward()

        data_vec = []

        # Gradient
        for i, layer in enumerate(keys):
            
            for param in layer.parameters():
                if len(param.shape) > 1:
                    break
            else:
                continue
            
            grad = param.grad_sample # [hidden_dim, input_dim]
            data_vec.append(grad.detach().view(bs, -1))
            
#         for i, li in enumerate(self.net_layer_ids):
#             grad = next(net[li].parameters()).grad_sample # [hidden_dim, input_dim]
#             data_vec.append(grad.detach().view(bs, -1))

#         for i, li in enumerate(self.head_layer_ids):
#             grad = next(head[li].parameters()).grad_sample
#             data_vec.append(grad.detach().view(bs, -1))

        data_vec = torch.cat(data_vec, dim = -1)
        return data_vec
    
    def patch_module(self, seq):
        for i in range(len(seq)):
            if isinstance(seq[i], nn.Linear):
                seq[i] = GradSampleModule(seq[i])
                
    def unpatch_module(self, seq):
        for i in range(len(seq)):
            if isinstance(seq[i], GradSampleModule):
                seq[i] = next(seq[i].children())
    
    def custom_query_step(self, budget, evidences, dataloader, model):
        '''Test method: spherical lerp'''
        
#         target_vec = [*self.net_sums, *self.head_sums]
        target_vec = [*self.sums]
        target_vec = [t.flatten() for t in target_vec]
        target_vec = torch.cat(target_vec)
        target_vec = target_vec.to(next(model.parameters()).device)

        # Create dummy network
        _, net, head, keys = model.getNets()
        net.load_state_dict(model.net_no_dropout.state_dict())
        head.load_state_dict(model.head.state_dict())
        
#         self.patch_module(net)
#         self.patch_module(head)
        net = GradSampleModule(net)
        head = GradSampleModule(head)
        net = net.to(target_vec.device)
        head = head.to(target_vec.device)
        
        sum_vec = torch.zeros_like(target_vec, device = target_vec.device)
        sum_norm = 0
        
        picked_indices = []
        
        sample_rate = 0.15

        for qid in range(budget):
            
            scores = []
            
            for batch in tqdm(dataloader, "Collecting query scores"):

                # skip some batches for speed-up
                if random.random() > sample_rate:
                    scores.append(np.ones(batch[0].shape[0],) * -99999999)
                    continue
                
                xs, _ = batch
                xs = xs.to(target_vec.device)

#                 batch_vec = []
                
#                 for bid in range(xs.shape[0]):

#                     data_vec = self.get_data_vec(model, xs[bid])
#                     batch_vec.append(data_vec)

#                 batch_vec = torch.stack(batch_vec, dim = 0)
                batch_vec = self.get_batch_vec(model, net, head, keys, xs)

                batch_norm = batch_vec.norm(dim = -1, keepdim = True)

                # Add previous length
                batch_vec = batch_vec + sum_vec[None, :]
                
                # Normalize
                batch_norm = batch_norm + sum_norm
                batch_vec = batch_vec / batch_vec.norm(dim = -1, keepdim = True) * batch_norm
                
                score = (batch_vec * target_vec[None, :]).sum(-1)
                
                scores.append((-score).detach().cpu().numpy())

            scores = np.concatenate(scores)
            idx = np.argsort(scores)[-1]
            picked_indices.append(idx) # Largest scores being picked
            
            # Retrieve information from picked index and update
            data_vec = self.get_data_vec(
                model,
                net,
                head,
                keys,
                dataloader.dataset[idx][0].to(target_vec.device)
            )
            sum_vec += data_vec
            sum_norm += data_vec.norm()
            
        return picked_indices

    
    
    
    
class MonteCarloBoundBatchedFast(MonteCarloBound):
    
    def custom_query_step(self, budget, evidences, dataloader, model):
        
        scores = []

        for batch in tqdm(dataloader, "Collecting query scores"):

            xs, _ = batch
            xs = xs.to(next(model.parameters()).device)

            for bid in range(xs.shape[0]):

                model.zero_grad()
                x = xs[bid].unsqueeze(0)

                feat = model.net_no_dropout(x)
                out = model.head(feat)
                fake_label = torch.argmax(out, dim = -1)

                loss = model.loss(out, fake_label)
                loss.backward()

                score = 0
                
                # Gradient
                for i, layer in enumerate(self.key_layers):
                    
                    for param in layer.parameters():
                        if len(param.shape) > 1:
                            break
                    else:
                        continue
                    
                    grad = param.grad # [hidden_dim, input_dim]
                    self.sums[i] = self.sums[i].to(grad.device)
                    score += (self.sums[i] * grad).sum().detach().cpu()

                scores.append(-score)

        sample_rate = 5.0
        num_random_pool = int(budget * sample_rate)
                
        scores = np.asarray(scores)
        indices = np.random.choice(np.argsort(scores)[-num_random_pool:], size = (budget, ), replace = False) # Largest scores being picked

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
        "mcgradient-batched": MonteCarloBoundBatched,
        "mcgradient-batched-fast": MonteCarloBoundBatchedFast
    }[name](shuffle_prop=shuffle_prop, reduction=reduction, **kwargs)
    return heuristic
