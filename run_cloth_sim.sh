
#!/bin/bash

# Exit if any command fails
set -e

echo "Building CUDA cloth simulation..."
mkdir -p build
cd build
cmake -DIMGUI_DIR=$HOME/imgui ..
make clean && make -j

echo "Running simulation..."

./cloth_sim