import wandb
from omegaconf import DictConfig, OmegaConf

def setup_wandb(cfg: DictConfig):
    """Initializes W&B with Hydra config."""
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.wandb.mode,
        # Allow resuming if job is preempted (using Slurm ID if available)
        resume="allow" 
    )
