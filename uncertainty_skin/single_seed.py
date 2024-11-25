import hydra
from src.single_seed import single_seed_train_and_test
from omegaconf import DictConfig

from src.utils.clearml_logger import ClearMLLogger

import uuid

@hydra.main(config_path="configs", config_name="config")
def main(config: DictConfig):
    
    single_seed_train_and_test(config = config)
    

if __name__ == "__main__":
    main()