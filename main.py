import hydra
from omegaconf import DictConfig
from scr.train import Trainer

@hydra.main(config_path="configs", config_name="config", version_base=None)
def train_pipeline(cfg: DictConfig):
    trainer = Trainer(cfg)
    trainer.pipeline()

if __name__ == "__main__":
    train_pipeline()