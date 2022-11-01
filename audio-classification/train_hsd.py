import argparse
import pytorch_lightning as pl
from hsd_exodus import ExodusNetwork
from hsd_slayer import SlayerNetwork
from hsd import HSD


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rand_seed", type=int, default=123, help="Provide a seed for random number generation")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size. Default: 128")
    parser.add_argument("--encoding_dim", type=int, default=100, help="Number of neurons in encoding layer. Default: 100")
    parser.add_argument(
        "--decoding_func",
        help="Use 'sum_loss', 'max_over_time'  or 'last_ts'.",
        type=str,
        default="max_over_time",
    )
    parser.add_argument("--hidden_dim", type=int, default=128, help="Number of neurons in hidden layer(s). Default: 128")
    parser.add_argument("--tau_mem", type=float, default=30.0, help="Membrane time constant in ms")
    parser.add_argument("--spike_threshold", type=float, default=1, help="Neuron firing threshold. Default: 1")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate during training. Default: 1e-3")
    parser.add_argument("--width_grad", type=float, default=1.0, help="Width of exponential surrogate gradient function. Default: 1.0")
    parser.add_argument("--scale_grad", type=float, default=1.0, help="Scaling of exponential surrogate gradient function. Default: 1.0")
    parser.add_argument("--n_hidden_layers", type=int, default=2, help="Number of hidden layers. Default: 2")
    parser.add_argument("--optimizer", type=str, default="adam", help="Define to use 'adam' or 'sgd'. Default: adam")
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    dict_args = vars(args)

    pl.seed_everything(args.rand_seed)

    data = HSD(
        batch_size=args.batch_size,
        encoding_dim=args.encoding_dim,
        num_workers=4,
        download_dir="./data",
    )

    slayer_model = SlayerNetwork(**dict_args, n_time_bins=250, output_dim=20)
    init_weights = slayer_model.state_dict()

    exodus_model = ExodusNetwork(**dict_args, init_weights=init_weights, output_dim=20)

    checkpoint_path = "models/checkpoints"

    for model_name, model in [["slayer", slayer_model], ["exodus", exodus_model]]:
        run_name = f"hsd/{model_name}/{args.tau_mem}/{args.scale_grad}/{args.optimizer}"
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="valid_acc",
            dirpath=checkpoint_path + "/" + run_name,
            filename="{run_name}-{step}-{epoch:02d}-{valid_loss:.2f}-{valid_acc:.2f}",
            save_top_k=1,
            mode="max",
        )

        logger = pl.loggers.TensorBoardLogger(save_dir="lightning_logs", name=run_name)
        trainer = pl.Trainer.from_argparse_args(
            args, logger=logger, callbacks=[checkpoint_callback], log_every_n_steps=20
        )

        trainer.logger.log_hyperparams(model.hparams)
        trainer.fit(model, data)

        print(f"Best model checkpoint path: {checkpoint_callback.best_model_path}")
