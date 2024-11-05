import hydra
from src.train import train
from src.utils.clearml_logger import ClearMLLogger
from omegaconf import DictConfig

import uuid


@hydra.main(config_path="configs", config_name="config")
def main(config: DictConfig):
    
    unique_id = uuid.uuid4()
    
    seed = config.dataset.seed
    
    logger = ClearMLLogger(
        project_name="ISIC_2024",
        task_name=f"{config.model.name}_{seed}_{config.model.loss_fun}_training_{unique_id}",
        offline=config.offline
    )

    
    train(config = config,
          seed = seed,
          logger = logger,
          unique_id=unique_id)
    
    
    

if __name__ == "__main__":
    main()