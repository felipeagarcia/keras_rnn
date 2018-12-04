# Recurrent Neural Network

This code implements a recurrent neural network using long short-term memory (lstm) architecture.

The data used for training and classification consists on raw inertial data, which is recommended to be normalized before
feeding the network.

## Prerequisites

To run this code, you will need Python3, tensorflow, numpy and sklearn.

To install the python libraries, use:

```
pip install tensorflow numpy sklearn
```

It is recomended to do this inside a virtual environment.

For a faster training, use tensorflow-gpu, here is a good guide to install it: https://github.com/williamFalcon/tensorflow-gpu-install-ubuntu-16.04


## Files

/data contains the datasets

/src contains the source code

## Running

To run the project, go to the src/ directory and type:

```
python main.py
```

This will train and evaluate the model.

To use the model in your project, just copy and paste the src/RNN.py file into your project and include it!