import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
import sinabs.layers as sl
import sinabs.activation as sa
from typing import Dict, Any
import sinabs
from torch.nn.utils import weight_norm


class Memory(nn.Sequential):
    def __init__(self, encoding_dim, output_dim, kw_args, backend):
        super().__init__(
            nn.Linear(encoding_dim, output_dim, bias=False),
            sinabs.exodus.layers.LIF(**kw_args) if backend == 'exodus' else sinabs.layers.LIF(**kw_args),
        )


class ExodusNetwork(pl.LightningModule):
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
        init_weights=None,
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
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

        if init_weights:
            self.network[0][0].weight.data = init_weights['linear_input.weight'].squeeze()
            for i in range(n_hidden_layers):
                self.network[i+1][0].weight.data = init_weights[f'linear_hidden.{i}.weight'].squeeze()
            self.network[-1].weight.data = init_weights['linear_output.weight'].squeeze()

        # self.activations = {}
        # for layer in self.spiking_layers:
        #     layer.register_forward_hook(self.save_activations)

    def forward(self, x):
        return self.network(x)

    def save_activations(self, module, input, output):
        self.activations[module] = output

    def training_step(self, batch, batch_idx):
        self.reset_states()
        x, y = batch  # x is Batch, Time, Channels
        y_hat = self(x)
        # firing_rate = torch.cat(list(self.activations.values())).mean()
        # self.log("firing_rate", firing_rate, prog_bar=True)
        if self.hparams.decoding_func == "sum_loss":
            y_decoded = y_hat.sum(1)
        if self.hparams.decoding_func == "max_over_time":
            y_decoded = y_hat.max(1)[0]
        elif self.hparams.decoding_func == "last_ts":
            y_decoded = y_hat[:, -1]
        loss = F.cross_entropy(y_decoded, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.reset_states()
        x, y = batch  # x is Batch, Time, Channels
        y_hat = self(x)
        if self.hparams.decoding_func == "sum_loss":
            y_decoded = y_hat.sum(1)
        if self.hparams.decoding_func == "max_over_time":
            y_decoded = y_hat.max(1)[0]
        elif self.hparams.decoding_func == "last_ts":
            y_decoded = y_hat[:, -1]
        loss = F.cross_entropy(y_decoded, y)
        prediction = y_decoded.argmax(1)
        self.log("valid_loss", loss, prog_bar=True)
        accuracy = (prediction == y).float().sum() / len(prediction)
        self.log("valid_acc", accuracy, prog_bar=True)

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            return torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
            )
        elif self.hparams.optimizer == "sgd":
            return torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
            )

    @property
    def sinabs_layers(self):
        return [
            layer
            for layer in self.network.modules()
            if isinstance(layer, sl.StatefulLayer)
        ]

    def reset_states(self):
        for layer in self.sinabs_layers:
            layer.reset_states()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        for name, parameter in checkpoint["state_dict"].items():
            # uninitialise states so that there aren't any problems
            # when loading the model from a checkpoint
            if "v_mem" in name or "activations" in name:
                checkpoint["state_dict"][name] = torch.zeros(
                    (0), device=parameter.device
                )
        return super().on_save_checkpoint(checkpoint)
