import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from src.data.datamodule import ISICDataModule
from src.models.cnn_model import CNN
from src.models.timm_model import TimmModel
from src.utils.clearml_logger import ClearMLLogger
from src.utils.utils import set_seed

@hydra.main(config_path="/repo/uncertainty_skin/uncertainty_skin/configs", config_name="config")
def train(config: DictConfig):
    set_seed(config.dataset.seed)
    data_module = ISICDataModule(config.dataset)
    if config.model.name == 'CNN':
        model = CNN(config.model)
    elif config.model.name.startswith('timm'):
        model = TimmModel(config.model)
    else:
        raise ValueError(f"Unknown model: {config.model.name}")

    logger = ClearMLLogger(project_name="ISIC_2024", task_name=f"{config.model.name}_training")
    trainer = pl.Trainer(**config.trainer, logger=logger)
    trainer.fit(model, data_module)

if __name__ == "__main__":
    train()