import torch
from torch import nn
from torch.nn import functional as F

import lightning as L
from .commons import Extension

class SeenClassesAccuracy(Extension):

    def setup(self, trainer, pl_module, stage):
        self.accuracy_seen = lambda p, y, o: ((torch.logical_and(p == y, o == 0)).float().sum() / (o == 0).float().sum()) if ((o == 0).float().sum() > 0) else None
        self.val_acc_seen = 0
        self.val_acc_seen_count = 0
    
    def on_validation_batch_end(self, trainer, plm, outputs, batch, batch_idx, dl_idx = 0):
        
        # Q: TODO: How do you obtain `output` from model that outputs nothing but a loss?
        # A: You don't...
        # breakpoint()

        i, x, y, o = batch

        logits = plm.cache.logits
        preds = torch.argmax(logits, dim=1)

        acc_seen = self.accuracy_seen(preds, y, o)

        if acc_seen is not None:
            self.val_acc_seen += acc_seen
            self.val_acc_seen_count += 1
            plm.log("val_acc_seen", acc_seen, prog_bar = True)
    
    def on_validation_epoch_end(self, trainer, plm):

        self.val_acc_seen = 0
        self.val_acc_seen_count = 0

    def get_val_acc_seen(self):
        
        if(self.val_acc_seen_count <= 0):
            return 0

        return self.val_acc_seen / self.val_acc_seen_count
