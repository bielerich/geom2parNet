

# Geometry to Parameter Mapping based on Neural Networks
This is a draft.

## About the Project
This project is part of my project thesis, in which I tried to find a way to find descriptive parameters from a given geometry represented by a point cloud. I wanted to answer the question: it is feasible to solve this problem using Deep Learning Neural Networks? 

I haven't worked with Neural Networks before, but I relatively quickly found out that working with point clouds as input parameters for NN is quite challenging. Some papers I found that adress this challenge are:


- [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/pdf/1612.00593.pdf)
- [PointNet++: Deep Hierarchical Feature Learning on
Point Sets in a Metric Space](https://arxiv.org/pdf/1706.02413.pdf)
- [PointCNN: Convolution On X -Transformed Points](https://arxiv.org/pdf/1801.07791.pdf)


For programming, I used [Tensorflow](https://www.tensorflow.org/) and [Keras](https://keras.io/). I followed this example of [Point cloud classification with PointNet](https://keras.io/examples/vision/pointnet/) for feeding the point cloud data into my NN. 


## Background Information


## Folder Structure
- *example_hull/*: Example for one hull geoemtry. 
This folder contains all necessary input data for executing the training as well as the output of one training validation procedure, consisting of 5x5 training runs. 

- *lib/*: contains all codes for reading in the data, executing the training and postprocessing.

## How to Execute
Prerequirements for executing the scripts are:

- python 3
- tensorflow version > 2.0
- keras
- numpy
- pandas
- scikit-learn

Navigate into *example_hull/*. Extract the input data *data/preparedData/DF_features*. Rhen run

> python trainNetwork.py

for executing the training. Run

> python postProcess.py

for postprocessing the result data.

## More Details
