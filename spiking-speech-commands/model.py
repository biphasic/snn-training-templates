import pytorch_lightning as pl
import sinabs
import sinabs.activation as sa
import sinabs.exodus.layers as el
import torch
import torch.nn as nn
import torchmetrics
from torch.nn import functional as F


class Memory(nn.Sequential):
    def __init__(self, encoding_dim, output_dim, kw_args):
        super().__init__(
            nn.Linear(encoding_dim, output_dim),
            el.LIF(**kw_args),
        )


class ExodusNet(pl.LightningModule):
    """EXODUS model using LIF layers. EXODUS can be installed from the following repository:
    https://github.com/synsense/sinabs-exodus and its API is the same as Sinabs layers which are
    documented here https://sinabs.readthedocs.io.

    Parameters:
        tau_mem: The time constant for the membrane potential.
        n_hidden_layers: How many hidden layers to use in the model, excluding input and output layer.
        spike_threshold: The spike threshold, defaults to 1.
        encoding_dim: How many neurons should be used in the input layer.
        hidden_dim: How many neurons should be used in a hidden layer.
        output_dim: How many neurons should be used in the output layer.
        decoding_func: You can choose how you want to optimise your network. Are you interested in the
                       sum of output activation over time? Then choose 'sum_loss'. If you're training
                       a streaming model and you're interested in the maximum activation over time,
                       choose 'max_over_time'. If you're interested in the activation at the last time
                       step, testing the network's memory capability, then select 'last_ts'.
        learning_rate: LR for the optimizer.
        width_grad: Surrogate gradient function width, defaults to 1.
        scale_grad: Surrogate gradient function scale, defaults to 1.
    """

    def __init__(
        self,
        tau_mem: float,
        n_hidden_layers: int,
        spike_threshold: float,
        encoding_dim: int,
        hidden_dim: int,
        output_dim: int,
        decoding_func: str = "max_over_time",
        learning_rate: float = 1e-3,
        width_grad: float = 1.0,
        scale_grad: float = 1.0,
        *args,
        **kw_args,
    ):
        super().__init__()
        self.save_hyperparameters()

        kw_args = dict(
            tau_mem=tau_mem,
            norm_input=False,
            spike_threshold=spike_threshold,
            spike_fn=sa.SingleSpike,
            reset_fn=sa.MembraneSubtract(),
            surrogate_grad_fn=sa.SingleExponential(
                grad_width=width_grad, grad_scale=scale_grad
            ),
        )

        self.network = nn.Sequential(
            Memory(encoding_dim, hidden_dim, kw_args),
            *[Memory(hidden_dim, hidden_dim, kw_args) for i in range(n_hidden_layers)],
            nn.Linear(hidden_dim, output_dim),
        )

        self.accuracy_metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=output_dim
        )

        self.decoder_dict = {
            "sum_loss": lambda y_hat: y_hat.sum(1),
            "max_over_time": lambda y_hat: y_hat.max(1)[0],
            "last_ts": lambda y_hat: y_hat[:, -1],
        }

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        sinabs.reset_states(self.network)
        x, y = batch  # x is Batch, Time, Channels
        y_hat = self(x.squeeze())
        y_decoded = self.decoder_dict[self.hparams.decoding_func](y_hat)
        loss = F.cross_entropy(y_decoded, y)
        self.log("loss/train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        sinabs.reset_states(self.network)
        x, y = batch  # x is Batch, Time, Channels
        y_hat = self(x.squeeze())
        y_decoded = self.decoder_dict[self.hparams.decoding_func](y_hat)
        loss = F.cross_entropy(y_decoded, y)
        self.log("loss/valid", loss, prog_bar=True)
        prediction = y_decoded.argmax(1)
        accuracy = self.accuracy_metric(prediction, y)
        self.log("accuracy/valid", accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        sinabs.reset_states(self.network)
        x, y = batch  # x is Batch, Time, Channels
        y_hat = self(x.squeeze())
        y_decoded = self.decoder_dict[self.hparams.decoding_func](y_hat)
        prediction = y_decoded.argmax(1)
        accuracy = self.accuracy_metric(prediction, y)
        self.log("accuracy/test", accuracy, prog_bar=True)
        self.log("hp_metric", accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )
