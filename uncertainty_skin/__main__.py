import os
import curses
import subprocess

project = 'uncertainty_skin'

# Function to display the main menu
def display_main_menu(stdscr, selected_row_idx):
    stdscr.clear()
    stdscr.addstr(0, 0, "================================")
    stdscr.addstr(1, 0, f"   Experiments for {project}")
    stdscr.addstr(2, 0, "================================")
    menu_items = ["Training", "Testing", "Training and Testing", "Setup ClearML Credentials", "Download Dataset", "Turn Logging Offline/Online", "Exit"]
    for idx, item in enumerate(menu_items):
        x = 0
        y = 3 + idx
        if idx == selected_row_idx:
            stdscr.attron(curses.color_pair(1))
            stdscr.addstr(y, x, item)
            stdscr.attroff(curses.color_pair(1))
        else:
            stdscr.addstr(y, x, item)
    stdscr.refresh()

# Function to display the Hydra experiment menu
def display_experiment_menu(stdscr, selected_row_idx):
    stdscr.clear()
    stdscr.addstr(0, 0, "================================")
    stdscr.addstr(1, 0, "   Select Hydra Experiment")
    stdscr.addstr(2, 0, "================================")
    config_files = [f for f in os.listdir(f"{project}/configs/experiment") if f.endswith(".yaml")]
    for idx, config in enumerate(config_files):
        x = 0
        y = 3 + idx
        if idx == selected_row_idx:
            stdscr.attron(curses.color_pair(1))
            stdscr.addstr(y, x, f"{idx + 1}. {os.path.splitext(config)[0]}")
            stdscr.attroff(curses.color_pair(1))
        else:
            stdscr.addstr(y, x, f"{idx + 1}. {os.path.splitext(config)[0]}")
    stdscr.refresh()
    return config_files

# Function to display the checkpoint menu
def display_checkpoint_menu(stdscr, selected_row_idx):
    stdscr.clear()
    stdscr.addstr(0, 0, "================================")
    stdscr.addstr(1, 0, "   Select Checkpoint")
    stdscr.addstr(2, 0, "================================")
    checkpoints = [f for f in os.listdir(f"{project}/checkpoints") if f.endswith(".ckpt")]
    for idx, checkpoint in enumerate(checkpoints):
        x = 0
        y = 3 + idx
        if idx == selected_row_idx:
            stdscr.attron(curses.color_pair(1))
            stdscr.addstr(y, x, f"{idx + 1}. {checkpoint}")
            stdscr.attroff(curses.color_pair(1))
        else:
            stdscr.addstr(y, x, f"{idx + 1}. {checkpoint}")
    stdscr.refresh()
    return checkpoints

# Function to setup ClearML credentials
def setup_clearml(stdscr):
    stdscr.clear()
    stdscr.addstr(0, 0, "Setting up ClearML credentials...")
    stdscr.refresh()
    CLEARML_API_HOST = stdscr.getstr(1, 0, 40).decode(encoding="utf-8")
    CLEARML_WEB_HOST = stdscr.getstr(2, 0, 40).decode(encoding="utf-8")
    CLEARML_API_ACCESS_KEY = stdscr.getstr(3, 0, 40).decode(encoding="utf-8")
    CLEARML_API_SECRET_KEY = stdscr.getstr(4, 0, 40).decode(encoding="utf-8")

    os.environ["CLEARML_API_HOST"] = CLEARML_API_HOST
    os.environ["CLEARML_WEB_HOST"] = CLEARML_WEB_HOST
    os.environ["CLEARML_API_ACCESS_KEY"] = CLEARML_API_ACCESS_KEY
    os.environ["CLEARML_API_SECRET_KEY"] = CLEARML_API_SECRET_KEY

    stdscr.addstr(5, 0, "ClearML credentials set successfully!")
    stdscr.refresh()
    stdscr.getch()

# Function to download the dataset
def download_dataset(stdscr):
    stdscr.clear()
    stdscr.addstr(0, 0, "Downloading the dataset...")
    stdscr.refresh()
    # Add your dataset download commands here
    # Example: subprocess.run(["wget", "<dataset_url>", "-O", "dataset.zip"])
    # Example: subprocess.run(["unzip", "dataset.zip"])
    stdscr.addstr(1, 0, "Dataset downloaded successfully!")
    stdscr.refresh()
    stdscr.getch()

# Function to turn logging offline/online
def turn_logging(stdscr):
    stdscr.clear()
    stdscr.addstr(0, 0, "Turn logging offline/online:")
    stdscr.addstr(1, 0, "1. Offline")
    stdscr.addstr(2, 0, "2. Online")
    stdscr.refresh()

    while True:
        choice = stdscr.getstr(3, 0, 1).decode(encoding="utf-8")
        if choice == "1":
            return "offline=True"
        elif choice == "2":
            return "offline=False"
        else:
            stdscr.addstr(4, 0, "Invalid choice. Please enter 1 or 2.")
            stdscr.refresh()

# Main function
def main(stdscr):
    curses.curs_set(0)
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
    current_row_idx = 0

    while True:
        display_main_menu(stdscr, current_row_idx)
        key = stdscr.getch()

        if key == curses.KEY_UP and current_row_idx > 0:
            current_row_idx -= 1
        elif key == curses.KEY_DOWN and current_row_idx < 6:
            current_row_idx += 1
        elif key == curses.KEY_ENTER or key in [10, 13]:
            if current_row_idx == 0:
                mode = "Training"
            elif current_row_idx == 1:
                mode = "Testing"
            elif current_row_idx == 2:
                mode = "Training and Testing"
            elif current_row_idx == 3:
                setup_clearml(stdscr)
                continue
            elif current_row_idx == 4:
                download_dataset(stdscr)
                continue
            elif current_row_idx == 5:
                logging_option = turn_logging(stdscr)
                continue
            elif current_row_idx == 6:
                stdscr.clear()
                stdscr.addstr(0, 0, "Exiting...")
                stdscr.refresh()
                break

            config_files = display_experiment_menu(stdscr, 0)
            experiment_row_idx = 0
            while True:
                key = stdscr.getch()
                if key == curses.KEY_UP and experiment_row_idx > 0:
                    experiment_row_idx -= 1
                elif key == curses.KEY_DOWN and experiment_row_idx < len(config_files) - 1:
                    experiment_row_idx += 1
                elif key == curses.KEY_ENTER or key in [10, 13]:
                    config_file = config_files[experiment_row_idx]
                    if mode == "Testing":
                        checkpoints = display_checkpoint_menu(stdscr, 0)
                        checkpoint_row_idx = 0
                        while True:
                            key = stdscr.getch()
                            if key == curses.KEY_UP and checkpoint_row_idx > 0:
                                checkpoint_row_idx -= 1
                            elif key == curses.KEY_DOWN and checkpoint_row_idx < len(checkpoints) - 1:
                                checkpoint_row_idx += 1
                            elif key == curses.KEY_ENTER or key in [10, 13]:
                                checkpoint = checkpoints[checkpoint_row_idx]
                                curses.endwin()  # Exit curses mode to restore terminal state
                                subprocess.run(["python", f"{project}/test.py", "+experiment=" + config_file, "model.checkpoint_path=" + checkpoint, logging_option])
                                stdscr.addstr(1, 0, "Experiment completed successfully!")
                                stdscr.refresh()
                                stdscr.getch()
                                break
                            display_checkpoint_menu(stdscr, checkpoint_row_idx)
                    else:
                        curses.endwin()  # Exit curses mode to restore terminal state
                        if mode == "Training":
                            subprocess.run(["python", f"{project}/train.py", "+experiment=" + config_file, logging_option])
                        elif mode == "Testing":
                            subprocess.run(["python", f"{project}/test.py", "+experiment=" + config_file, logging_option])
                        elif mode == "Training and Testing":
                            subprocess.run(["python", f"{project}/multi_seed.py", "+experiment=" + config_file, logging_option])
                        stdscr.addstr(1, 0, "Experiment completed successfully!")
                        stdscr.refresh()
                        stdscr.getch()
                    break
                display_experiment_menu(stdscr, experiment_row_idx)

if __name__ == "__main__":
    curses.wrapper(main)