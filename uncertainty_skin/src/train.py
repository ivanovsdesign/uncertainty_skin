import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from src.data.datamodule import ISICDataModule, ChestDataModule
from src.models.cnn_model import CNN
from src.models.timm_model import TimmModel
from src.utils.clearml_logger import ClearMLLogger
from src.utils.utils import set_seed
import torch
import os
import uuid

from src.utils.metrics import plot_pca
from src.utils.utils import extract_features

def create_model(config):
    if config.model.name == 'CNN':
        model = CNN(config)
    elif config.model.name in config.timm_models:
        model = TimmModel(config)
    else:
        raise ValueError(f"Unknown model: {config.model.name}")
    return model

def train(config: DictConfig,
          seed: int,
          logger: callable,
          unique_id: str):
    
    set_seed(seed)

    
    config.dataset.seed = seed
    
    model_slug = f'{config.model.name}_{seed}_{config.model.loss_fun}_{unique_id}'

    os.makedirs('checkpoints', exist_ok=True)

    data_module = ISICDataModule(config.dataset)
    model = create_model(config)
    
    # Extract embeddings before training
    
    data_module.setup()
    
    features_before, labels_before = extract_features(model, data_module.train_dataloader())

    checkpoint_callback = ModelCheckpoint(
        save_weights_only=True, mode="min", monitor="val_loss", dirpath='checkpoints/',
        filename=f'{model_slug}_' + '{epoch}'
    )

    callbacks = [
        checkpoint_callback,
        LearningRateMonitor("epoch")
    ]

    trainer = pl.Trainer(max_epochs=config.trainer.max_epochs,
                         logger=logger,
                         callbacks=callbacks,
                         deterministic=True)
    
    trainer.fit(model, data_module)

    model.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path, weights_only=True)['state_dict'])

    os.makedirs('models', exist_ok=True)
    save_path = f"{config.model.name}_{seed}_training_{unique_id}.pt"
    torch.save(model, os.path.join('models', save_path))

    print(f'Model trained on seed {seed} saved')
    os.environ['TRAINED_MODEL'] = save_path

    with open(f'{os.path.join(os.getcwd(), 'trained_paths.csv')}', 'a') as file:
        file.write(f'{seed}, {config.model.name}, {trainer.checkpoint_callback.best_model_path}\n')

    original_dir = get_original_cwd()
    os.makedirs(os.path.join(original_dir, 'checkpoints'), exist_ok=True)
    os.system(f"cp {trainer.checkpoint_callback.best_model_path} {os.path.join(original_dir, 'checkpoints')}")

    features_after, labels_after = extract_features(model, data_module.train_dataloader())
    
    print(features_after)
    
    try:
        plot_pca(features_before, labels_before, f'Class Distribution Before Training', model_slug)
        plot_pca(features_after, labels_after, f'Class Distribution After Training', model_slug)    
    except: 
        print('Cannot extract features and plot embeddings')

    return trainer.checkpoint_callback.best_model_path

if __name__ == "__main__":
    train()