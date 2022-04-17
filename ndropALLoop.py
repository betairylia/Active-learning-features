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
        with torch.no_grad():
            out = []
            for _ in range(self.inference_iteration):
                out.append(self.parent_module.predict_step(batch, batch_idx))

        # BaaL expects a shape [num_samples, num_classes, num_iterations]
        return torch.stack(out).permute((1, 2, 0))
    
class ndrop_ActiveLearningLoop(ActiveLearningLoop):
    
    def on_run_start(self, *args, **kwargs) -> None:
        # assert isinstance(self.trainer.datamodule, ActiveLearningDataModule)
        if self._datamodule_state_dict is not None:
            self.trainer.datamodule.load_state_dict(self._datamodule_state_dict)
        self.trainer.predict_loop._return_predictions = True
        self._lightning_module = self.trainer.lightning_module
        self._model_state_dict = deepcopy(self._lightning_module.state_dict())
        # self.inference_model = InferenceMCDropoutTask(self._lightning_module, self.inference_iteration)
        self.inference_model = ndrop_InferenceMCDropoutTask(self._lightning_module, self.inference_iteration)
        
        # We need to set-up AL datasets eariler.
        self.trainer.datamodule.setup()
        
    def advance(self, *args, **kwargs) -> None:

        self.progress.increment_started()

        if self.trainer.datamodule.has_labelled_data:
            self.fit_loop.run()

        if self.trainer.datamodule.has_test:
            self._reset_testing()
            metrics = self.trainer.test_loop.run()
            if metrics:
                self.trainer.logger.log_metrics(metrics[0], step = self.progress.current.completed) # Use current step for AL
            
            # Also log current number of labelled samples
            self.trainer.logger.log_metrics(
                {'labelled': self.trainer.datamodule._dataset.n_labelled}, 
                step = self.progress.current.completed
            )

        if self.trainer.datamodule.has_unlabelled_data:
            self._reset_predicting()
            probabilities = self.trainer.predict_loop.run()
            self.trainer.datamodule.label(probabilities=probabilities)
        else:
            raise StopIteration

        self._reset_fitting()
        self.progress.increment_processed()