
# Create build directory
rm -rf build
mkdir -p build

# Configure the project with CMake
cmake -S . -B build \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_INSTALL_PREFIX=$(pwd)/build/install && \
  cmake --build build -j $(nproc) && \
  cmake --install build

# Create virtual environment in the install directory for the python module.
python3 -m venv build/install/convexa




# install the python module into the virtual environment.
source ./build/install/convexa/bin/activate > out.txt

export CMAKE_PREFIX_PATH=`pwd`/build/install
export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:/usr/local/lib/python3.10/dist-packages/pybind11 \
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/build/install/lib
python3 -m pip install ./src

# run test
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/build/install/lib
python validation/precision_valid.py

# Deactivate the virtual environment
deactivate






