import os
import argparse
import yaml
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from lightning_module import EQEstimatorLightning
from dataset_pipeline.dataset import create_dataloaders

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../conf/config.yaml")
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # 1. Set seed
    pl.seed_everything(config['seed'])
    
    # 2. Setup Data
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        split=(config['data']['train_split'], config['data']['val_split'], config['data']['test_split'])
    )
    
    # 3. Init Model
    model = EQEstimatorLightning(config)
    
    # 4. Callbacks & Logger
    wandb_logger = WandbLogger(project=config['experiment_name'], log_model="all")
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val/total_loss",
        dirpath="checkpoints",
        filename="eq-model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val/total_loss",
        min_delta=0.001,
        patience=10,
        verbose=True,
        mode="min"
    )
    
    # 5. Trainer
    trainer = pl.Trainer(
        max_epochs=config['trainer']['max_epochs'],
        accelerator=config['trainer']['accelerator'],
        devices=config['trainer']['devices'],
        precision=config['trainer']['precision'],
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback, LearningRateMonitor(logging_interval='step')],
        fast_dev_run=config['trainer']['fast_dev_run'],
        log_every_n_steps=config['trainer']['log_every_n_steps'],
    )
    
    # 6. Fit
    trainer.fit(model, train_loader, val_loader)
    
    # 7. Test
    trainer.test(model, test_loader)

if __name__ == "__main__":
    main()
