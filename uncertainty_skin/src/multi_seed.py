import hydra
from omegaconf import DictConfig
import pandas as pd
from src.train import train
from src.test import test

from src.utils.clearml_logger import ClearMLLogger

import uuid

def multi_seed_train_and_test(config: DictConfig):
    
    unique_id = uuid.uuid4()
    
    logger = ClearMLLogger(
        project_name="ISIC_2024",
        task_name=f"{config.model.name}_multiseed_{config.model.loss_fun}_{unique_id}",
        offline=config.offline
    )
    
    seeds = [42, 0, 3, 9, 17]
    all_summary_dfs = []

    for seed in seeds:
        checkpoint_path = train(config, seed, logger, unique_id)
        summary_df = test(config, seed, checkpoint_path, logger, unique_id)
        all_summary_dfs.append(summary_df)

    if len(all_summary_dfs) > 1:
        combined_summary_df = pd.concat(all_summary_dfs)
        metrics_std = combined_summary_df.iloc[:, 4:].std()
        metrics_error = combined_summary_df.iloc[:, 4:].sem()

        # Save combined metrics
        combined_summary_df.to_csv(f"{config.model.name}_combined_summary.csv", index=False)
        metrics_std.to_csv(f"{config.model.name}_metrics_std.csv")
        metrics_error.to_csv(f"{config.model.name}_metrics_error.csv")
    
    logger._task.upload_artifact('Run summary', combined_summary_df)
    logger._task.upload_artifact('Metrics_std', metrics_std)
    logger._task.upload_artifact('Metrics_error', metrics_error)
    
    logger.report_table(
        title="Run summary", 
        series="Metrics",
        table_plot=combined_summary_df
    )
    
    logger.report_table(
        title="Metrics STD", 
        series="Metrics",
        table_plot=metrics_std
    )
    
    logger.report_table(
        title="Metrics Error", 
        series="Metrics",
        table_plot=metrics_error
    )
    
    logger._task.close()
    

if __name__ == "__main__":
    multi_seed_train_and_test()