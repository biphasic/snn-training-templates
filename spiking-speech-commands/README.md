# Spiking speech commands

This is a fully functional experiment setup to train a model on the Heidelberg spiking speech commands dataset, with a focus on clean code, speed and modularity. The repo is organized in Pytorch Lightning manner into data (ssc.py), model (model.py) and the training entry point via command line (train.py). 

## Install requirements 
```
pip install -r requirements.txt
```

## Run the training script
You can use all of Pytorch Lightning's [training flags](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags) that are available, but a basic run is started with:
```
python train.py --accelerator=gpu --devices=1 --max_epochs=100
```

While the model is training, you can follow the metrics being logged to Tensorboard! You can call `tensorboard --logdir ./lightning_logs` from the terminal or simply use VS code's integrated [plugin](https://code.visualstudio.com/docs/datascience/pytorch-support#_tensorboard-integration)!

Any issues you have please let me know.