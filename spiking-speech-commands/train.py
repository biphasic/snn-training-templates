import argparse

import pytorch_lightning as pl
from model import ExodusNet
from ssc import SSC

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument(
        "--rand_seed",
        type=int,
        default=123,
        help="Provide a seed for random number generation",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size. Default: 128"
    )
    parser.add_argument(
        "--encoding_dim",
        type=int,
        default=100,
        help="Number of neurons in encoding layer. Default: 100",
    )
    parser.add_argument(
        "--decoding_func",
        help="Use 'sum_loss', 'max_over_time'  or 'last_ts'.",
        type=str,
        default="max_over_time",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Number of neurons in hidden layer(s). Default: 128",
    )
    parser.add_argument(
        "--tau_mem", type=float, default=30.0, help="Membrane time constant in ms"
    )
    parser.add_argument(
        "--spike_threshold",
        type=float,
        default=1,
        help="Neuron firing threshold. Default: 1",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate during training. Default: 1e-3",
    )
    parser.add_argument(
        "--width_grad",
        type=float,
        default=1.0,
        help="Width of exponential surrogate gradient function. Default: 1.0",
    )
    parser.add_argument(
        "--scale_grad",
        type=float,
        default=1.0,
        help="Scaling of exponential surrogate gradient function. Default: 1.0",
    )
    parser.add_argument(
        "--n_hidden_layers",
        type=int,
        default=2,
        help="Number of hidden layers. Default: 2",
    )
    args = parser.parse_args()
    dict_args = vars(args)

    pl.seed_everything(args.rand_seed)

    dataset = SSC(**dict_args)
    model = ExodusNet(**dict_args, output_dim=35)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="accuracy/valid",
        filename="step={step}-epoch={epoch}-valid_loss={loss/valid}-valid_acc={accuracy/valid}",
        auto_insert_metric_name=False,
        save_top_k=1,
        mode="max",
    )
    run_name = f"ssc/{args.tau_mem}_tau_mem"
    logger = pl.loggers.TensorBoardLogger(save_dir="lightning_logs", name=run_name)
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback],
        logger=logger,
    )

    trainer.fit(model, dataset)
    trainer.test(model, dataset)
