import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from src.data.datamodule import ISICDataModule
from src.models.cnn_model import CNN
from src.models.timm_model import TimmModel
from src.utils.clearml_logger import ClearMLLogger
from src.utils.utils import set_seed

import torch

import os

import uuid

def create_model(config):
    if config.model.name == 'CNN':
        model = CNN(config.model)
    elif config.model.name in config.timm_models:
        model = TimmModel(config.model)
    else:
        raise ValueError(f"Unknown model: {config.name}")
    
    return model

@hydra.main(config_path="/repo/uncertainty_skin/uncertainty_skin/configs", config_name="config")
def train(config: DictConfig):
    set_seed(config.dataset.seed)

    unique_id = uuid.uuid4()

    os.makedirs(config.trainer.checkpoint_path)

    data_module = ISICDataModule(config.dataset)
    model = create_model(config)

    checkpoint_callback = ModelCheckpoint(
            save_weights_only=True, mode="min", monitor="val_loss", dirpath=config.trainer.checkpoint_path,
            filename=f'{config.model.name}_{config.dataset.seed}_{config.model.loss_fun}_{unique_id}_' + '{epoch}'
        )

    callbacks=[
        checkpoint_callback,  # Save the best checkpoint based on the min val_loss recorded. Saves only weights and not optimizer
        LearningRateMonitor("epoch")
    ]

    logger = ClearMLLogger(project_name="ISIC_2024",
                           task_name=f"{config.model.name}_{config.dataset.seed}_{config.model.loss_fun}_training_{unique_id}",
                           offline=config.offline)
    trainer = pl.Trainer(max_epochs=config.trainer.max_epochs, logger=logger, callbacks=callbacks)
    trainer.fit(model, data_module)

    model.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path, weights_only=True)['state_dict'])

    os.makedirs('models', exist_ok=True)
    save_path = f"{config.model.name}_{config.dataset.seed}_training_{unique_id}.pt"
    torch.save(model, os.path.join('models', save_path))

    print('Model saved')
    os.environ['TRAINED_MODEL'] = save_path

if __name__ == "__main__":
    train()