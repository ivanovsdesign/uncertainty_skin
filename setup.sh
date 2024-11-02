#!/bin/bash

# Define the environment name
ENV_NAME="uncertainty_isic"

# Define the Python version (optional)
PYTHON_VERSION="3.12.4"

handle_error() {
    echo "Error: $1" >&2
    exit 1
}

# Create the Conda environment
echo "Creating Conda environment..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Activate the environment
echo "Activating the environment..."
conda activate $ENV_NAME || handle_error "Failed to activate environment $ENV_NAME."

echo "Installing requirements..."
pip install -r requirements.txt || handle_error "Failed to install requirements\n Try to install manually."

# Verify the environment was created and activated
if [[ $CONDA_DEFAULT_ENV == $ENV_NAME ]]; then
    echo "Environment $ENV_NAME has been successfully created and activated."
else
    handle_error "Failed to activate environment $ENV_NAME."
fi