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
    menu_items = [f"{idx + 1}. {os.path.splitext(config)[0]}" for idx, config in enumerate(config_files)] + ["Go Back"]
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
    return config_files, menu_items

# Function to display the checkpoint menu
def display_checkpoint_menu(stdscr, selected_row_idx, start_idx=0):
    stdscr.clear()
    stdscr.addstr(0, 0, "================================")
    stdscr.addstr(1, 0, "   Select Checkpoint")
    stdscr.addstr(2, 0, "================================")
    os.makedirs('checkpoints', exist_ok=True)
    checkpoints = [f for f in os.listdir(f"checkpoints") if f.endswith(".ckpt")]
    if not checkpoints:
        stdscr.addstr(3, 0, "No checkpoints found. Please train the model first.")
        stdscr.refresh()
        stdscr.getch()
        return None, 0, []
    max_y, max_x = stdscr.getmaxyx()
    menu_items = [f"{idx + start_idx + 1}. {checkpoint}" for idx, checkpoint in enumerate(checkpoints[start_idx:start_idx + max_y - 4])] + ["Go Back"]
    for idx, item in enumerate(menu_items):
        x = 0
        y = 3 + idx
        if y >= max_y:
            break
        if idx == selected_row_idx:
            stdscr.attron(curses.color_pair(1))
            stdscr.addstr(y, x, item)
            stdscr.attroff(curses.color_pair(1))
        else:
            stdscr.addstr(y, x, item)
    stdscr.refresh()
    return checkpoints, start_idx, menu_items

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

# Function to display the logging menu
def display_logging_menu(stdscr, selected_row_idx):
    stdscr.clear()
    stdscr.addstr(0, 0, "================================")
    stdscr.addstr(1, 0, "   Turn Logging Offline/Online")
    stdscr.addstr(2, 0, "================================")
    logging_options = ["Offline", "Online", "Go Back"]
    for idx, option in enumerate(logging_options):
        x = 0
        y = 3 + idx
        if idx == selected_row_idx:
            stdscr.attron(curses.color_pair(1))
            stdscr.addstr(y, x, option)
            stdscr.attroff(curses.color_pair(1))
        else:
            stdscr.addstr(y, x, option)
    stdscr.refresh()
    return logging_options

# Main function
def main(stdscr):
    
    logging_option_selected = "offline=False"
    
    os.system('clear')
    
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
                logging_options = display_logging_menu(stdscr, 0)
                logging_row_idx = 0
                while True:
                    key = stdscr.getch()
                    if key == curses.KEY_UP and logging_row_idx > 0:
                        logging_row_idx -= 1
                    elif key == curses.KEY_DOWN and logging_row_idx < len(logging_options) - 1:
                        logging_row_idx += 1
                    elif key == curses.KEY_ENTER or key in [10, 13]:
                        if logging_options[logging_row_idx] == "Go Back":
                            break
                        logging_option = logging_options[logging_row_idx]
                        if logging_option == "Offline":
                            logging_option_selected = "offline=True"
                        else:
                            logging_option_selected = "offline=False"
                        break
                    display_logging_menu(stdscr, logging_row_idx)
                continue
            elif current_row_idx == 6:
                stdscr.clear()
                stdscr.addstr(0, 0, "Exiting...")
                stdscr.refresh()
                break

            config_files, experiment_menu_items = display_experiment_menu(stdscr, 0)
            experiment_row_idx = 0
            while True:
                key = stdscr.getch()
                if key == curses.KEY_UP and experiment_row_idx > 0:
                    experiment_row_idx -= 1
                elif key == curses.KEY_DOWN and experiment_row_idx < len(experiment_menu_items) - 1:
                    experiment_row_idx += 1
                elif key == curses.KEY_ENTER or key in [10, 13]:
                    if experiment_menu_items[experiment_row_idx] == "Go Back":
                        break
                    config_file = config_files[experiment_row_idx]
                    if mode == "Testing":
                        checkpoints, start_idx, checkpoint_menu_items = display_checkpoint_menu(stdscr, 0)
                        if checkpoints is None:
                            continue
                        checkpoint_row_idx = 0
                        while True:
                            key = stdscr.getch()
                            if key == curses.KEY_UP and checkpoint_row_idx > 0:
                                checkpoint_row_idx -= 1
                            elif key == curses.KEY_DOWN and checkpoint_row_idx < len(checkpoint_menu_items) - 1:
                                checkpoint_row_idx += 1
                            elif key == curses.KEY_PPAGE:  # Page Up
                                start_idx = max(0, start_idx - (max_y - 4))
                                checkpoints, start_idx, checkpoint_menu_items = display_checkpoint_menu(stdscr, checkpoint_row_idx, start_idx)
                            elif key == curses.KEY_NPAGE:  # Page Down
                                start_idx = min(len(checkpoints) - (max_y - 4), start_idx + (max_y - 4))
                                checkpoints, start_idx, checkpoint_menu_items = display_checkpoint_menu(stdscr, checkpoint_row_idx, start_idx)
                            elif key == curses.KEY_ENTER or key in [10, 13]:
                                if checkpoint_menu_items[checkpoint_row_idx] == "Go Back":
                                    break
                                checkpoint = checkpoints[checkpoint_row_idx]
                                curses.endwin()  # Exit curses mode to restore terminal state
                                subprocess.run(["python", f"{project}/main_test.py", "+experiment=" + config_file, logging_option_selected])
                                stdscr.addstr(1, 0, "Experiment completed successfully!")
                                stdscr.refresh()
                                stdscr.getch()
                                break
                            display_checkpoint_menu(stdscr, checkpoint_row_idx, start_idx)
                    else:
                        curses.endwin()  # Exit curses mode to restore terminal state
                        if mode == "Training":
                            subprocess.run(["python", f"{project}/main_train.py", "+experiment=" + config_file, logging_option_selected])
                        elif mode == "Training and Testing":
                            subprocess.run(["python", f"{project}/multi_seed.py", "+experiment=" + config_file, logging_option_selected])
                        stdscr.addstr(1, 0, "Experiment completed successfully!")
                        stdscr.refresh()
                        stdscr.getch()
                    break
                display_experiment_menu(stdscr, experiment_row_idx)

if __name__ == "__main__":
    curses.wrapper(main)