import lightning as L
from utils import ImageMosaicSQ

import torch
from torch.nn import functional as F

class Extension(L.Callback):

    def __init__(self, args):
        self.args = args
    
    def on_test_epoch_start(self, t, p):
        return self.on_validation_epoch_start(t, p)

    def on_test_epoch_end(self, t, p):
        return self.on_validation_epoch_end(t, p)

    def on_test_batch_start(self, t, p, b, bi, di = 0):
        return self.on_validation_batch_start(t, p, b, bi, di)

    def on_test_batch_end(self, t, p, o, b, bi, di = 0):
        return self.on_validation_batch_end(t, p, o, b, bi, di)

class DatasetVisualizer(Extension):

    def __init__(self, args):
        super().__init__(args)
        self.visualized = False

    def on_validation_batch_start(self, trainer, plm, batch, batch_idx, dataloader_idx = 0):

        if not self.visualized:

            i, x, y, o = batch

            plm.logger.log_image(
                key = "test-set",
                images = [ImageMosaicSQ(x)],
                caption = ["".join(["1" if _o == 1 else "0" for _o in o.detach().cpu()])]
            )

            self.visualized = True

class LogitsStatstics(Extension):

    def entropy(self, logits):

        probs = F.softmax(logits, dim = -1)
        model_prediction = probs.mean(0)
        entropy = -torch.sum(model_prediction * torch.log(model_prediction + 1e-8), dim = 1)
        return entropy

    def on_validation_batch_end(self, trainer, plm, o, b, bid, did = 0):

        l = plm.cache.logits
        plm.log("logitStats/max", l.max(dim=-1)[0].mean())
        plm.log("logitStats/mean", l.mean())
        plm.log("logitStats/stddev", l.std(dim=-1).mean())
        plm.log("logitStats/entropy", self.entropy(l.unsqueeze(0)).mean())
    
    def on_test_batch_end(self, t, p, o, b, bi, di = 0):
        return self.on_validation_batch_end(t, p, o, b, bi, di)
