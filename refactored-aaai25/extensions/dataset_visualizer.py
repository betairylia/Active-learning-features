import lightning as L
from utils import ImageMosaicSQ

class DatasetVisualizer(L.Callback):

    def __init__(self, args):
        super().__init__()
        self.visualized = False
        self.args = args
    
    def on_validation_batch_start(self, trainer, plm, batch, batch_idx, dataloader_idx = 0):

        if not self.visualized:

            i, x, y, o = batch

            plm.logger.log_image(
                key = "test-set",
                images = [ImageMosaicSQ(x)],
                caption = ["".join(["1" if _o == 1 else "0" for _o in o.detach().cpu()])]
            )

            self.visualized = True
