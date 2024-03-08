import torch
from .base import SimpleModel

class InjectTest(SimpleModel):

    def extra_init(self, reference_dl):

        self.reference_dl = torch.utils.data.DataLoader(
            PickTwoDataset(reference_dl.dataset, 1, 2),
            batch_size = 16,
            shuffle = False,
            num_workers = 0
        )

    def forward(self, x):

        if self.la is not None:
            pred = self.la(x, link_approx='probit')
        else:
            print("[WARNING] LaplaceRedux in running without LA (currently using --model=default as a fallback)!")
            logits = self.head(self.net(x))
            pred = F.softmax(logits, dim = 1)

        entropy = -torch.sum(pred_prob * torch.log(pred_prob + 1e-8), dim = 1)

        return logits, entropy

    def on_validation_epoch_start(self):

        torch.set_grad_enabled(True)

        # Assemble net & head
        model = nn.Sequential(self.net, self.head)

        # Fit the LA with self.reference_dl
        self.la = Laplace(model, 'classification',
             subset_of_weights='all',
             hessian_structure='diag')
        self.la.fit(self.reference_dl)

        # Perhaps we don't optimize the prior here
        # la.optimize_prior_precision(method='CV', val_loader=val_loader)

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        torch.set_grad_enabled(False)