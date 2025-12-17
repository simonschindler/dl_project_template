import hydra
from omegaconf import DictConfig
import torch
import wandb

# Import from our local package
from my_model.utils.logging import setup_wandb
from my_model.models.module import SimpleModel

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # 1. Setup Logging
    setup_wandb(cfg)
    
    print(f"Training on device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print(f"Configuration:\n{cfg}")

    # 2. Initialize Model
    model = SimpleModel(num_classes=cfg.model.num_classes)
    
    # 3. Dummy Training Loop
    # (In real usage, this would be your Lightning/PyTorch loop)
    for epoch in range(cfg.epochs):
        # Simulate metric logging
        loss = 1.0 / (epoch + 1)
        wandb.log({"loss": loss, "epoch": epoch})
        print(f"Epoch {epoch}: Loss {loss}")

    wandb.finish()

if __name__ == "__main__":
    main()
