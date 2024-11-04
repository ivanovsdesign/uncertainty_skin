#!/bin/bash

# Function to display the main menu
display_main_menu() {
    clear
    echo "================================"
    echo "   Interactive Menu"
    echo "================================"
    echo "1. Training"
    echo "2. Testing"
    echo "3. Training and Testing"
    echo "4. Setup ClearML Credentials"
    echo "5. Download Dataset"
    echo "6. Exit"
    echo "================================"
}

# Function to display the Hydra experiment menu
display_experiment_menu() {
    clear
    echo "================================"
    echo "   Select Hydra Experiment"
    echo "================================"
    local i=1
    for config in configs/experiment/*.yaml; do
        echo "$i. $(basename "$config" .yaml)"
        CONFIG_FILES[$i]=$config
        ((i++))
    done
    echo "================================"
}

# Function to setup ClearML credentials
setup_clearml() {
    clear
    echo "Setting up ClearML credentials..."
    read -p "Enter ClearML API Server: " CLEARML_API_HOST
    read -p "Enter ClearML Web Server: " CLEARML_WEB_HOST
    read -p "Enter ClearML API Access Key: " CLEARML_API_ACCESS_KEY
    read -p "Enter ClearML API Secret Key: " CLEARML_API_SECRET_KEY

    export CLEARML_API_HOST
    export CLEARML_WEB_HOST
    export CLEARML_API_ACCESS_KEY
    export CLEARML_API_SECRET_KEY

    echo "ClearML credentials set successfully!"
    read -n 1 -s -r -p "Press any key to continue..."
}

# Function to download the dataset
download_dataset() {
    clear
    echo "Downloading the dataset..."
    # Add your dataset download commands here
    # Example: wget <dataset_url> -O dataset.zip && unzip dataset.zip
    echo "Dataset downloaded successfully!"
    read -n 1 -s -r -p "Press any key to continue..."
}

# Main script
while true; do
    display_main_menu
    PS3="Choose an option (1-6): "
    select MAIN_MENU_OPTION in "Training" "Testing" "Training and Testing" "Setup ClearML Credentials" "Download Dataset" "Exit";
        case $MAIN_MENU_OPTION in
            "Training")
                MODE="Training"
                break
                ;;
            "Testing")
                MODE="Testing"
                break
                ;;
            "Training and Testing")
                MODE="Training and Testing"
                break
                ;;
            "Setup ClearML Credentials")
                setup_clearml
                continue 2
                ;;
            "Download Dataset")
                download_dataset
                continue 2
                ;;
            "Exit")
                echo "Exiting..."
                exit 0
                ;;
            *)
                echo "Invalid option. Exiting..."
                exit 1
                ;;
        esac
    done

    display_experiment_menu
    PS3="Choose an experiment (1-${#CONFIG_FILES[@]}): "
    select EXPERIMENT_OPTION in "${!CONFIG_FILES[@]}"; do
        if [[ -z $EXPERIMENT_OPTION ]]; then
            echo "Invalid experiment option. Exiting..."
            exit 1
        fi
        CONFIG_FILE=${CONFIG_FILES[$EXPERIMENT_OPTION]}
        break
    done

    case $MODE in
        "Training")
            echo "Running training with config: $CONFIG_FILE"
            python train.py +experiment=$CONFIG_FILE
            ;;
        "Testing")
            echo "Running testing with config: $CONFIG_FILE"
            python test.py +experiment=$CONFIG_FILE
            ;;
        "Training and Testing")
            echo "Running training and testing with config: $CONFIG_FILE"
            python train.py +experiment=$CONFIG_FILE
            python test.py +experiment=$CONFIG_FILE
            ;;
    esac

    echo "Experiment completed successfully!"
    read -n 1 -s -r -p "Press any key to continue..."
done