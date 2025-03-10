#!/bin/sh

if [ $# -ne ]; then
  echo "No Args required."
  exit 1
fi

# Check if the virtual environment already exists
if [ -d ".venv" ]; then
  echo "Virtual environment '.venv' already exists."
  echo "Activating..."
fi

# Create the virtual environment
python3 -m venv ".venv"

echo "Virtual environment '.venv' created successfully."

# Activate the virtual environment (optional)
source ".venv/bin/activate"

echo "Virtual environment activated."

echo "Installing dependencies."
# TODO: For some reason this does not work
python3 -m pip install -r requirements.txt


SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR" || exit

cmake -S . -B build
cd build
make

cd ../validation
