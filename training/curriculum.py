"""
Curriculum learning callback for multi-type EQ estimation.

Manages staged training with increasing difficulty:
1. Peaking-only warmup
2. Multi-type with high type loss weight
3. Full difficulty with spectral consistency
4. Real-world fine-tuning
"""
import torch
import pytorch_lightning as pl


class CurriculumCallback(pl.Callback):
    """
    Lightning callback that manages curriculum learning stages.
    Adjusts loss weights, parameter ranges, and Gumbel-Softmax temperature.
    """

    def __init__(self, config):
        super().__init__()
        self.stages = config.get("curriculum", {}).get("stages", [])
        self.current_stage_idx = 0

    def _compute_stage(self, epoch):
        """Determine which curriculum stage we're in based on epoch."""
        cumulative = 0
        for i, stage in enumerate(self.stages):
            cumulative += stage.get("epochs", 10)
            if epoch < cumulative:
                return i
        return len(self.stages) - 1

    def on_train_epoch_start(self, trainer, pl_module):
        stage_idx = self._compute_stage(trainer.current_epoch)

        if stage_idx != self.current_stage_idx:
            self.current_stage_idx = stage_idx
            stage = self.stages[stage_idx]
            print(f"\n[Curriculum] Entering stage {stage_idx}: {stage.get('name', 'unnamed')}")

            # Update loss weights
            lambda_type = stage.get("lambda_type", 0.5)
            pl_module.criterion.lambda_type = lambda_type

            # Update Gumbel-Softmax temperature
            temperature = stage.get("gumbel_temperature", 0.5)
            pl_module.model.param_head.gumbel_temperature.fill_(temperature)

            # Update learning rate if specified
            lr_scale = stage.get("learning_rate_scale", None)
            if lr_scale is not None:
                for pg in trainer.optimizers[0].param_groups:
                    pg["lr"] = pl_module.config["model"]["learning_rate"] * lr_scale
