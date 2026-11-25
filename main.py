import warnings
warnings.filterwarnings("ignore")

from argparse import ArgumentParser
import os

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.strategies.ddp import DDPStrategy

from dataset.data_api import LitDataModule
from model.model_api import LitModel
from misc.utils import load_cfg, merge_args_cfg

def main(args):
    L.seed_everything(args.seed, workers=True)
    dm = LitDataModule(hparams=args)
    model = LitModel(hparams=args)
    monitor = "val_mpjpe"
    filename = f"{args.model_name}" + "-{epoch}-{val_mpjpe:.4f}"

    callbacks = [
        ModelCheckpoint(
            monitor=monitor,
            dirpath=os.path.join('logs', args.exp_name, args.version),
            filename=filename,
            save_top_k=1,
            save_last=True,
            mode='min'),
        RichProgressBar(refresh_rate=20),
        LearningRateMonitor(logging_interval='epoch'),
    ]

    logger = TensorBoardLogger(save_dir='logs',
                               name=args.exp_name,
                               version=args.version)
    logger.log_hyperparams(args)

    trainer = L.Trainer(
        fast_dev_run=args.dev,
        logger=[logger, CSVLogger(save_dir='logs', name=args.exp_name, version=args.version)],
        max_epochs=args.epochs,
        # Prediction returns materialized tensors, so keep it to a single device.
        devices=1 if args.predict else args.gpus,
        accelerator="gpu",
        sync_batchnorm=args.sync_batchnorm,
        num_nodes=args.num_nodes,
        gradient_clip_val=args.clip_grad,
        strategy=DDPStrategy(find_unused_parameters=False) if args.strategy == 'ddp' else args.strategy,
        callbacks=callbacks,
        precision=args.precision,
        benchmark=args.benchmark
    )

    if bool(args.test):
        trainer.test(model, datamodule=dm)
    elif bool(args.predict):
        predictions = trainer.predict(model, datamodule=dm, return_predictions=True)
        save_path = os.path.join('logs', args.exp_name, args.version, f'{args.model_name}_predictions.pt')
        print(f'Saving predictions to {save_path}')
        torch.save(predictions, save_path)
    else:
        trainer.fit(model, datamodule=dm, ckpt_path=args.resume)
        if args.dev == 0:
            trainer.test(ckpt_path="best", datamodule=dm)

if __name__ == '__main__':
    parser = ArgumentParser(description="Train, test, or predict with a config-driven Lightning run.")
    parser.add_argument('-c', '--cfg', type=str, default='cfg/test.yaml')
    parser.add_argument('-g', '--gpus', type=int, default=None,
                        help='number of gpus to use (default: all available)')
    parser.add_argument('-d', "--dev", type=int, default=0, help='fast_dev_run for debug')
    parser.add_argument('-n', "--num_nodes", type=int, default=1)
    parser.add_argument('-w', "--num_workers", type=int, default=4)
    parser.add_argument('-b', "--batch_size", type=int, default=2048)
    parser.add_argument('-e', "--batch_size_eva", type=int, default=1000, help='batch_size for evaluation')
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument("--model_ckpt_dir", type=str, default="./model_ckpt/")
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_data_path', help='override the test dataset pickle (use with --test_split)')
    parser.add_argument('--test_split', help='split in --test_data_path')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None, help='resume model, optimizer, scheduler, and epoch')
    parser.add_argument('--exp_name', type=str, default='fasternet')
    parser.add_argument("--version", type=str, default="0")

    args = parser.parse_args()
    cli_seed = args.seed
    cfg = load_cfg(args.cfg)
    args = merge_args_cfg(args, cfg)
    args.seed = cli_seed if cli_seed is not None else getattr(cfg, 'seed', 3407)
    if args.test_data_path or args.test_split:
        if not (args.test and args.test_data_path and args.test_split):
            parser.error('use --test, --test_data_path and --test_split together')
        args.test_dataset['params'].update(data_path=args.test_data_path, split=args.test_split)

    main(args)
