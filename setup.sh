#!/bin/bash

# Stop if any command fails
set -e

ENV_NAME="venv"
PYTHON_VERSION="3.11"
VENV_DIR="./$ENV_NAME"
SOCCERTRACK_DATASET_URL="https://storage.googleapis.com/kaggle-data-sets/2481051/4977082/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250317%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250317T170738Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=aaa6d70f85bd329a01c7c95061916ecd5152b3bf6e0d158e1946cac6e8aa5c3b42fa0da535b90991a66a8fb19612743305b18b0da0c098006a56d3a49a7850d3202494c0c32b8852e5291b6b414e7ba8dd8e5903904a0137024a97418cb1b8a46462a29341a7b0303151a08335989856d931fc4b982d67a063f96e121dac0bb2ac8c0e09d1f1b1c5edf6ed17fd03de7e194793234ff7f4d2f2145a4a374340d686a01c860cebb686aa9ffad58571211ce0644a6307eb6729b9bd5a224a94d0ddb66e0b683b996e18cd8b70b8a2eccaa0aadfb4f12ce21dbdb4a0215dd5479352615df1e94c396d61983e69bf34f98437c0a18d9040bee133b2f80e3e9d0f5f74"
SOCCERTRACK_DATASET_ZIP_NAME="soccertrack.zip"
SOCCERTRACK_FOLDER="./soccertrack"

echo "---------------------------------------"
echo "Setting up Python $PYTHON_VERSION virtual environment: $ENV_NAME"
echo "---------------------------------------"

# Check Python 3.10 availability
if ! python3.11 --version &>/dev/null; then
    echo "Python 3.11 is not installed!"
    echo "Install it first. Example for Ubuntu:"
    echo "sudo add-apt-repository ppa:deadsnakes/ppa -y"
    echo "sudo apt update"
    echo "sudo apt install python3.11 python3.11-venv python3.11-dev -y"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment in $VENV_DIR"
python3.11 -m venv "$VENV_DIR"

# Activate virtual environment
source "$VENV_DIR/bin/activate"

echo "Activated virtual environment: $VENV_DIR"

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Downloading SoccerTrack data"
wget "$SOCCERTRACK_DATASET_URL" -O "$SOCCERTRACK_DATASET_ZIP_NAME"

echo "Extracting ZIP file..."
unzip "$SOCCERTRACK_DATASET_ZIP_NAME" -d "$SOCCERTRACK_FOLDER"

echo "Removing ZIP file.."
rm "$SOCCERTRACK_DATASET_ZIP_NAME"

echo "---------------------------------------"
echo "Setup Complete!"
echo "To activate your environment later, run:"
echo "source $VENV_DIR/bin/activate"
echo "---------------------------------------"
