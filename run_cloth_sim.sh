
#!/bin/bash

# Exit if any command fails
set -e

echo "Building CUDA cloth simulation..."
mkdir -p build
mkdir -p output
cd build
cmake -DIMGUI_DIR=$HOME/imgui ..
make -j

echo "Running simulation..."

./cloth_sim