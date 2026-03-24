# CowIDentifier
Individual Cow IDentification part of MultiCamCows2024 Project, more info can be found [here](https://phoenix4582.github.io/MultiCamCows2024.github.io/)

# Setup
We strongly recommend using Anaconda for setting up environment. Once you installed Anaconda, simply type:
```
conda env create --name your_envname --file=lightning_id.yaml
```
to build up the environment

# Additional resources
This repository mostly utilises PyTorch and PyTorch Lightning modules. Additional documentation and tutorial can be found at:

PyTorch Vision: https://pytorch.org/vision/stable/index.html

PyTorch Lightning: https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html

# Notes
Both sections now added KFold capabilities and updated with the training configs of the paper.

## For the supervised section:
It's recommended to run the 'supervised_main.py' with 'supervised_config.yaml'; or the kfold counterparts ('main_kfold_supervised.py', 'config_kfold_fused.yaml'); over the sole 'main.py' and 'config.yaml' which contains the oldest model pipelines.

## For the self-supervised section:
The folder now holds all the code needed for standalone training, with KFold capability, instead of code segments that needed to be replaced within the files of Supervised folder, which brings confusion.
