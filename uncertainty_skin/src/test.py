import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from src.data.datamodule import ISICDataModule
from src.models.cnn_model import CNN
from src.models.timm_model import TimmModel
from src.utils.clearml_logger import ClearMLLogger
from src.utils.utils import set_seed
import torch
import os

# Set memory management environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

@hydra.main(config_path="/repo/uncertainty_skin/uncertainty_skin/configs/", config_name="config")
def test(config: DictConfig):
    set_seed(config.dataset.seed)
    data_module = ISICDataModule(config.dataset)
    data_module.setup()

    print(config.model.name)

    if config.model.name == 'CNN':
        model = CNN.load_from_checkpoint(config.model.checkpoint_path, config=config)
    elif config.model.name in config.timm_models:
        model = TimmModel.load_from_checkpoint(config.model.checkpoint_path, config=config)
    else:
        raise ValueError(f"Unknown model: {config.model.name}")

    logger = ClearMLLogger(project_name="ISIC_2024", task_name=f"{config.model.name}_testing", offline=config.offline)
    trainer = pl.Trainer(**config.trainer, logger=logger)

    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    test()