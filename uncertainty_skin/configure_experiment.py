import yaml
import os

def prompt_user(prompt, default=None):
    if default:
        return input(f"{prompt} [{default}]: ") or default
    else:
        return input(f"{prompt}: ")

def configure_experiment():
    # Prompt the user for each parameter
    model_name = prompt_user("Enter model name", "CNN")
    loss_fun = prompt_user("Enter loss function", "TM+UANLL")
    checkpoint_path = prompt_user("Enter checkpoint path", "/repo/uncertainty_skin/outputs/2024-11-02/13-25-48/checkpoints/CNN_42_TM+UANLL_5e5183ea-9d19-493f-a1ae-799dac919ce8_epoch=0.ckpt")
    img_size = prompt_user("Enter image size", "32")

    # Create the YAML configuration
    config = {
        "defaults": [
            {"_self_": None}
        ],
        "model": {
            "name": model_name,
            "loss_fun": loss_fun,
            "checkpoint_path": checkpoint_path
        },
        "dataset": {
            "img_size": int(img_size)
        }
    }

    # Write the YAML configuration to a file
    project_dir = "uncertainty_skin"
    config_dir = os.path.join(project_dir, "configs", "experiment")
    os.makedirs(config_dir, exist_ok=True)

    config_file_name = f"{model_name}_{loss_fun}.yaml"
    config_file_path = os.path.join(config_dir, config_file_name)

    with open(config_file_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    print(f"Experiment configuration saved to {config_file_path}")

if __name__ == "__main__":
    configure_experiment()