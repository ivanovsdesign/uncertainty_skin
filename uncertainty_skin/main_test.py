import hydra
from src.test import test
from src.utils.clearml_logger import ClearMLLogger
from omegaconf import DictConfig


@hydra.main(config_path="configs", config_name="config")
def main(config: DictConfig):
    
    logger = ClearMLLogger(project_name="ISIC_2024",
                           task_name=f"{config.model.name}_testing",
                           offline=config.offline)
    
    seed = config.dataset.seed
    checkpoint_path = config.model.checkpoint_path
    
    test(config = config,
         seed = seed,
         checkpoint_path = checkpoint_path,
         logger = logger)
    

if __name__ == "__main__":
    main()