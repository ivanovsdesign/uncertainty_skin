import hydra
from src.test import test
from src.utils.clearml_logger import ClearMLLogger
from omegaconf import DictConfig
import uuid
import logging 
import pandas as pd

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

@hydra.main(config_path="configs", config_name="config")
def main(config: DictConfig):
    
    all_predictions_df = pd.DataFrame()
    
    for checkpoint_path in config.model.checkpoint_path:
        logging.info(f'Checkpoint path: {checkpoint_path}')
        #checkpoint_path = config.model.checkpoint_path
        
        unique_id = checkpoint_path.split('_')[4]
        
        seed = int(checkpoint_path.split('_')[2])
        logging.info(f'Seed: {seed}')
        
        logger = ClearMLLogger(project_name="ISIC_2024",
                            task_name=f"{config.model.name}_{seed}_{config.model.loss_fun}_testing",
                            offline=config.offline)
        
        _, predictions_df = test(config = config,
                                seed = seed,
                                checkpoint_path = checkpoint_path,
                                logger = logger,
                                unique_id=unique_id)
        
        all_predictions_df = pd.concat([all_predictions_df, predictions_df])
                            
        logger._task.close()
        

if __name__ == "__main__":
    main()