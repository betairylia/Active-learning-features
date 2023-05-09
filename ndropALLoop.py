# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from copy import deepcopy
from pytorch_lightning.utilities.exceptions import MisconfigurationException

import flash
from flash.core.utilities.imports import _BAAL_AVAILABLE

if _BAAL_AVAILABLE:
    from baal.bayesian.dropout import _patch_dropout_layers
    
from flash.image.classification.integrations.baal import (
    ActiveLearningDataModule,
    ActiveLearningLoop
)

# TODO: Make this out of flash
class ndrop_InferenceMCDropoutTask(flash.Task):
    def __init__(self, module: flash.Task, inference_iteration: int):
        super().__init__()
        self.parent_module = module
        self.trainer = module.trainer
        changed = _patch_dropout_layers(self.parent_module)
        if not changed and inference_iteration > 1:
            print("The model does not contain dropout layer, inference_iteration has been set to 1.")
            inference_iteration = 1
        self.inference_iteration = inference_iteration

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        
        net = None
        
        with torch.no_grad():
            
            out = []
            fin = []
            
            for _ in range(self.inference_iteration):
                (logits, features, net) = self.parent_module.query_step(batch, batch_idx)
                out.append(logits)
                fin.append(features)

        # BaaL expects a shape [num_samples, num_classes, num_iterations]
        return (
            torch.stack(out).permute((1, 2, 0)), # [N_sample, dim, N_iter]
            torch.stack(fin).permute((1, 2, 0)), # [N_sample, dim, N_iter]
            net
        )
    
class ndrop_ActiveLearningLoop(ActiveLearningLoop):
    
    def __init__(self, daug_trials, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.daug_trials = daug_trials
    
    def on_run_start(self, *args, **kwargs) -> None:
        # assert isinstance(self.trainer.datamodule, ActiveLearningDataModule)
        if self._datamodule_state_dict is not None:
            self.trainer.datamodule.load_state_dict(self._datamodule_state_dict)
        self.trainer.predict_loop._return_predictions = True
        self._lightning_module = self.trainer.lightning_module
        self._model_state_dict = deepcopy(self._lightning_module.state_dict())
        # self.inference_model = InferenceMCDropoutTask(self._lightning_module, self.inference_iteration)
        # self.inference_model = ndrop_InferenceMCDropoutTask(self._lightning_module, self.inference_iteration)
        self.inference_model = self._lightning_module
        
        # We need to set-up AL datasets eariler.
        self.trainer.datamodule.setup()
        
    def combine_evidences(self, evidences):
        
        result = []
        
        # for i in range(N_batches)
        for i in range(len(evidences[0])):
            
            # Skip if no evidence
            if evidences[0][i] is None:
                continue
            
            row = []
            
            # num of tensors included in evidences[0][i]
            for j in range(len(evidences[0][i])):
                
                col = []
                # for evidence in evidences "num of prediction loops / per DA"
                for evidence in evidences:
                    col.append(evidence[i][j])

                if len(col) > 0:
                    row.append(torch.cat(col, -1))
            
            result.append(row)
        
        return result
        
    def advance(self, *args, **kwargs) -> None:

        self.progress.increment_started()

        if self.trainer.datamodule.has_labelled_data:
            self.fit_loop.run()

        if self.trainer.datamodule.has_test:
            
            self._reset_testing()
            metrics = self.trainer.test_loop.run()
            
            if metrics:
                metrics_AL = metrics[0]
                metrics_AL = {('AL/%s'%k): metrics_AL[k] for k in metrics_AL}
                
#                 stddrop, stdndrop = self.inference_model.EvalStddevEpisode()
#                 if stddrop is not None:
#                     metrics_AL['AL/std-Dropout'] = stddrop
#                     metrics_AL['AL/std-MonteCarlo'] = stdndrop
                
                # Also log current number of labelled samples
                metrics_AL['AL/labelled'] = self.trainer.datamodule._dataset.n_labelled
                self.trainer.logger.log_metrics(metrics_AL, step = self.progress.current.completed) # Use current step for AL
                # self.trainer.logger.log_metrics(metrics_AL) # Use current step for AL

        # Handle querying
        if self.trainer.datamodule.has_unlabelled_data:
            
            # Reset for new query round
            self._reset_predicting()
            self.trainer.datamodule.prediction_reset()
            
            # Collect evidences
            evidences = []
            
            # Collect several trials for each unlabeled data-point (considering data augmentation)
            for i in range(self.daug_trials):
                evidence = self.trainer.predict_loop.run()
                evidences.append(evidence)
            
            # Combine them along the "N_iter" axis (originally for MC-dropout)
            evidences = self.combine_evidences(evidences)
            
            qIm = self.trainer.datamodule.label(
                evidences=evidences,
                net=self._lightning_module,
                getImg = True
            )
            self.trainer.logger.log_image(key = "AL/queriesVis", images = [qIm], step = self.progress.current.completed)
        else:
            raise StopIteration

        self._reset_fitting()
        self.progress.increment_processed()
        
# TODO: Move to other place
def ReweightFeatures(features, weight):
    
    '''
    Reweights features accroading to weight.
    features: [N_batches (Nb), dim]
    weight:   [dim,]
    '''
    
    # Apply weight sign and sort features and abs weights
    features_signed = features * (torch.sign(weights))
    features_sorted, features_order = torch.sort(features_signed)
    weights_sorted = torch.abs(weights)[features_order] # => [Nb, dim]
    
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
    uniformed = torch.linspace(start = 0, end = weights_total[0], steps = weights.shape[0])#.unsqueeze(0) * weights_total.unsqueeze(1)
    uniformed = uniformed.unsqueeze(0).repeat(weights_cumsum.shape[0], 1)

    # Perform binary search to find interpolation ends
    searched_results = torch.searchsorted(weights_cumsum, uniformed)
    searched_results[:, 0] = 1 # Remove first 0's 
    
    # Linear interpolation: starts[ <------------ interp --> ] ends
    starts = torch.gather(features_sorted, -1, searched_results - 1)
    ends = torch.gather(features_sorted, -1, torch.minimum(searched_results, torch.LongTensor([features_sorted.shape[-1] - 1], device = features_sorted.device)))

    # Linear interpolation: obtain interp from both weight ends
    weights_s = torch.gather(weights_cumsum, -1, searched_results - 1)
    weights_e = torch.gather(weights_cumsum, -1, searched_results)
    interp = (uniformed - weights_s) / (weights_e - weights_s)
    
    # Do the interpolation
    result = starts + (ends - starts) * interp
    return result
