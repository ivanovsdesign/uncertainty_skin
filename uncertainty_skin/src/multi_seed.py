import hydra
from omegaconf import DictConfig
import pandas as pd
from src.train import train
from src.test import test

@hydra.main(config_path="/repo/uncertainty_skin/uncertainty_skin/configs", config_name="config")
def multi_seed_train_and_test(config: DictConfig):
    seeds = [42, 0, 3, 9, 17]
    all_summary_dfs = []

    for seed in seeds:
        checkpoint_path = train(config, seed)
        summary_df = test(config, seed, checkpoint_path)
        all_summary_dfs.append(summary_df)

    if len(all_summary_dfs) > 1:
        combined_summary_df = pd.concat(all_summary_dfs)
        metrics_std = combined_summary_df.std()
        metrics_error = combined_summary_df.sem()

        # Save combined metrics
        combined_summary_df.to_csv(f"{config.model.name}_combined_summary.csv", index=False)
        metrics_std.to_csv(f"{config.model.name}_metrics_std.csv")
        metrics_error.to_csv(f"{config.model.name}_metrics_error.csv")

if __name__ == "__main__":
    multi_seed_train_and_test()