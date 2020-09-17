#!/bin/sh

##--------- MintPy ------------------##
export MINTPY_HOME=${HOME}/tools/MintPy
export PYTHONPATH=${PYTHONPATH}:${MINTPY_HOME}
export PATH=${PATH}:${MINTPY_HOME}/mintpy

##--------- PyAPS -------------------##
export PYAPS_HOME=${HOME}/tools/PyAPS
export PYTHONPATH=${PYTHONPATH}:${PYAPS_HOME}

git clone https://github.com/insarlab/MintPy.git $MINTPY_HOME
git clone https://github.com/yunjunz/pyaps3.git $PYAPS_HOME/pyaps3

# install dependencies with conda
conda config --add channels conda-forge
conda install --yes --file $MINTPY_HOME/docs/conda.txt
pip install git+https://github.com/tylere/pykml.git