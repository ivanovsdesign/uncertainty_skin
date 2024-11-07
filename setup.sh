#!/bin/bash

# Define the environment name
ENV_NAME="uncertainty_skin"

# Define the Python version (optional)
PYTHON_VERSION="3.12.4"

handle_error() {
    echo "Error: $1" >&2
    exit 1
}


# Check if the environment already exists
if conda env list | grep -q "^$ENV_NAME\s"; then
    echo "Environment $ENV_NAME already exists. Activating it..."

    # Activate the environment
    echo "Activating the environment..."
    conda activate $ENV_NAME || handle_error "Failed to activate environment $ENV_NAME."

else
    # Create the Conda environment
    echo "Creating Conda environment..."
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y || handle_error "Failed to create environment $ENV_NAME." && conda activate $ENV_NAME

    # Activate the environment
    echo "Activating the environment..."
    conda activate $ENV_NAME || handle_error "Failed to activate environment $ENV_NAME."

    echo "Installing requirements..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt || handle_error "Failed to install requirements\n Try to install manually."

fi

if [ ! -f clearml.sh ]; then
    echo "Creating clearml.sh..."
    cp clearml_example.sh clearml.sh
fi

source clearml.sh

# Verify the environment was created and activated
if [[ $CONDA_DEFAULT_ENV == $ENV_NAME ]]; then
    echo "Environment $ENV_NAME has been successfully activated."
else
    handle_error "Failed to activate environment $ENV_NAME."
fi