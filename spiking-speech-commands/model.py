import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
import sinabs.activation as sa
from typing import Dict, Any
import sinabs
import torchmetrics

class Memory(nn.Sequential):
    def __init__(self, encoding_dim, output_dim, kw_args, backend):
        super().__init__(
            nn.Linear(encoding_dim, output_dim),
            sinabs.exodus.layers.LIF(**kw_args) if backend == 'exodus' else sinabs.layers.LIF(**kw_args),
        )


class ExodusNet(pl.LightningModule):
    def __init__(
        self,
        tau_mem,
        n_hidden_layers,
        spike_threshold,
        encoding_dim,
        hidden_dim,
        output_dim,
        decoding_func='max_over_time',
        learning_rate=1e-3,
        width_grad=1.,
        scale_grad=1.,
        backend='exodus',
        **kw_args,
    ):
        super().__init__()
        self.save_hyperparameters(ignore='init_weights')

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
            Memory(encoding_dim, hidden_dim, kw_args, backend),
            *[
                Memory(hidden_dim, hidden_dim, kw_args, backend)
                for i in range(n_hidden_layers)
            ],
            nn.Linear(hidden_dim, output_dim),
        )

        self.accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=output_dim)

        self.decoder_dict = {
            "sum_loss": lambda y_hat: y_hat.sum(1),
            "max_over_time": lambda y_hat: y_hat.max(1)[0],
            "last_ts": lambda y_hat: y_hat[:, -1],
        }

    def forward(self, x):
        return self.network(x)

    def save_activations(self, module, input, output):
        self.activations[module] = output

    def training_step(self, batch, batch_idx):
        sinabs.reset_states(self.network)
        x, y = batch  # x is Batch, Time, Channels
        y_hat = self(x)
        y_decoded = self.decoder_dict[self.hparams.decoding_func](y_hat)
        loss = F.cross_entropy(y_decoded, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        sinabs.reset_states(self.network)
        x, y = batch  # x is Batch, Time, Channels
        y_hat = self(x)
        y_decoded = self.decoder_dict[self.hparams.decoding_func](y_hat)
        loss = F.cross_entropy(y_decoded, y)
        self.log("valid_loss", loss, prog_bar=True)
        prediction = y_decoded.argmax(1)
        accuracy = self.accuracy_metric(prediction, y)
        self.log("valid_acc", accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        sinabs.reset_states(self.network)
        x, y = batch  # x is Batch, Time, Channels
        y_hat = self(x)
        y_decoded = self.decoder_dict[self.hparams.decoding_func](y_hat)
        prediction = y_decoded.argmax(1)
        accuracy = self.accuracy_metric(prediction, y)
        self.log("test_acc", accuracy, prog_bar=True)


    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        for name, parameter in checkpoint["state_dict"].items():
            # uninitialise states so that there aren't any problems
            # when loading the model from a checkpoint
            if "v_mem" in name or "activations" in name:
                checkpoint["state_dict"][name] = torch.zeros(
                    (0), device=parameter.device
                )
        return super().on_save_checkpoint(checkpoint)
