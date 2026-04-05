import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
import yaml
import sys
sys.path.append('.')
from training.lightning_module import EQEstimatorLightning

def main():
    # Load config
    with open('conf/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    config['trainer']['fast_dev_run'] = True
        
    print("Creating dummy in-memory dataset...")
    # 1. Dummy Data: MelSpec [B, C, F, T], Targets [B, P], Audio [B, Samples]
    num_samples = 32
    # The CNN expects [B, 1, 128, X] - for 3 seconds of 22050 audio at hop_length=256, frames ~= 259
    frames = 259
    mel_specs = torch.randn(num_samples, 1, 128, frames)
    # The target params for 6 bands: 3 params each = 18 elements
    target_params = torch.rand(num_samples, 18)
    # Target waveform for MS-STFT loss dummy pass
    target_audio = torch.randn(num_samples, int(3.0 * 22050))
    
    dataset = TensorDataset(mel_specs, target_params, target_audio)
    loader = DataLoader(dataset, batch_size=config['data']['batch_size'])
    
    # 2. Model
    model = EQEstimatorLightning(config)
    
    # 3. Trainer
    trainer = pl.Trainer(
        fast_dev_run=True,
        accelerator='cpu', # CPU for CI test runs
        enable_checkpointing=False,
        logger=False
    )
    
    # 4. Fit & Test
    print("Starting fast_dev_run verification...")
    trainer.fit(model, loader, loader)
    trainer.test(model, loader)
    print("Lightning Verification Passed ✔")

if __name__ == "__main__":
    main()
