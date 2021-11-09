from phinet_pl.phinet import PhiNet

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.autograd import Variable
import pytorch_lightning as pl
from torchinfo import summary
import numpy as np
import torch

import click


@click.command()
@click.argument('exp_id', type=str)
@click.argument('data_path', type=str)
def main(exp_id, data_path):
    wandb_logger = WandbLogger(log_model=True)
    if exp_id == "cifar10":
        from examples.cifar10 import PNCifar10, Cifar10Dataset

        mod = PNCifar10()
        data_module = Cifar10Dataset(data_path)
        

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
                monitor='val/loss',
                dirpath='./ckp',
                filename='models-{epoch:02d}-{val/loss:.2f}',
                save_top_k=3,
                mode='min')
        
        lr_logger = LearningRateMonitor()

        trainer = pl.Trainer(gpus=1,
                            max_epochs=150,
                            callbacks=[checkpoint_callback, lr_logger],
                            logger=wandb_logger)

        trainer.fit(model=mod, datamodule=data_module)
        trainer.test(datamodule=data_module, verbose=1)
    
    else:
        pn = PhiNet(res=32, B0=9, alpha=0.25, beta=1.3, t_zero=6, squeeze_excite=True, h_swish=True, include_top=True).to("cpu").eval()    
        
        rand_x = Variable(torch.Tensor(np.random.rand(1, 3, 32, 32))).to("cpu")

        print(pn.forward(rand_x).shape)
        summary(pn, input_data=rand_x, col_names = ["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], verbose=1, device="cpu")


if __name__ == "__main__":
    main()