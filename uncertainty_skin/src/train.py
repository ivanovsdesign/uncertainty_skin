import hydra
from hydra.utils import get_original_cwd
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
        model = CNN(config)
    elif config.model.name in config.timm_models:
        model = TimmModel(config)
    else:
        raise ValueError(f"Unknown model: {config.model.name}")
    return model

@hydra.main(config_path="/repo/uncertainty_skin/uncertainty_skin/configs", config_name="config")
def train(config: DictConfig, seed: int):
    set_seed(seed)
    unique_id = uuid.uuid4()
    
    config.dataset.seed = seed

    os.makedirs('checkpoints', exist_ok=True)

    data_module = ISICDataModule(config.dataset)
    model = create_model(config)

    checkpoint_callback = ModelCheckpoint(
        save_weights_only=True, mode="min", monitor="val_loss", dirpath='checkpoints/',
        filename=f'{config.model.name}_{seed}_{config.model.loss_fun}_{unique_id}_' + '{epoch}'
    )

    callbacks = [
        checkpoint_callback,
        LearningRateMonitor("epoch")
    ]

    logger = ClearMLLogger(
        project_name="ISIC_2024",
        task_name=f"{config.model.name}_{seed}_{config.model.loss_fun}_training_{unique_id}",
        offline=config.offline
    )

    trainer = pl.Trainer(max_epochs=config.trainer.max_epochs, logger=logger, callbacks=callbacks)
    trainer.fit(model, data_module)

    model.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path, weights_only=True)['state_dict'])

    os.makedirs('models', exist_ok=True)
    save_path = f"{config.model.name}_{seed}_training_{unique_id}.pt"
    torch.save(model, os.path.join('models', save_path))

    print(f'Model trained on seed {seed} saved')
    os.environ['TRAINED_MODEL'] = save_path

    with open(f'{os.path.join(os.getcwd(), 'trained_paths.csv')}', 'a') as file:
        file.write(f'{seed}, {config.model.name}, {trainer.checkpoint_callback.best_model_path}\n')

    os.chdir(get_original_cwd())
    os.makedirs('checkpoints', exist_ok=True)
    os.system(f'cp {trainer.checkpoint_callback.best_model_path} checkpoints/')

    return trainer.checkpoint_callback.best_model_path

if __name__ == "__main__":
    train()