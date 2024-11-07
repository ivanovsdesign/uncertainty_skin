import hydra
from src.test import test
from src.utils.clearml_logger import ClearMLLogger
from omegaconf import DictConfig
import uuid


@hydra.main(config_path="configs", config_name="config")
def main(config: DictConfig):

    unique_id = uuid.uuid4()
    
    seed = config.dataset.seed
    
    logger = ClearMLLogger(project_name="ISIC_2024",
                           task_name=f"{config.model.name}_{seed}_{config.model.loss_fun}_testing",
                           offline=config.offline)

    checkpoint_path = config.model.checkpoint_path
    
    test(config = config,
         seed = seed,
         checkpoint_path = checkpoint_path,
         logger = logger,
         unique_id=unique_id)
    
    logger._task.close()
    

if __name__ == "__main__":
    main()