# Self-Driving Car Engineer Nanodegree
# Deep Learning
## Project: Behavioral Cloning

### Overview

This project is created as the 3rd project of Self-Driving Car Nanodegree Program by Udacity. 

The goal is to drive a car autonomously in a simulator using a deep neuronal network based on the training data on the simulator of human driving behavior.

Bellow is the result on the training track and testing track.

![](https://github.com/etmaxwellsdemon/CarND-Behavioral-Cloning/blob/master/Track1.gif)

### Dependencies

This project requires **Python 3.5** and the following Python libraries installed:

- [Jupyter](http://jupyter.org/)
- [NumPy](http://www.numpy.org/)
- [TensorFlow](http://tensorflow.org)
- numpy
- flask-socketio
- eventlet
- pillow
- h5py
- keras
- OpenCV

### Dataset

1. At first, the dataset was created by keyboard(arrow keys). However, it cannot generate smooth data for training. I'll wait for my new Xbox controller to create training data again.

2. Then we tried the dataset created by [Udacity officially](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)

### Preprocessing
1. We convert the RGB to YUV according to the [paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

![](https://github.com/etmaxwellsdemon/CarND-Behavioral-Cloning/blob/master/YUV.png)

2. We resampled the image by the factor of 4, from 320x160 to 80x40

3. Then we cropped top 13 pixals as it's useless for training.

4. For left and right camera images, we've added a fixed shift angle of -0.2 to the right camera and 0.2 to the left camera.

I've also tried to histogram the Y channel, the result is not better than the un-histogramed one.

Also, I've generate mirror images with steering angles to create more training dataset, but the result is not that good.

### Network Structure
The network structure and the training parameter are both included in the model.py file and the Jupyter Notebook file(behavioral cloning.ipynb)
 

The structure is identical to the [Nividia paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf), which is listed bellow:


Layer (type)|Output Shape|Param #|Connected to                     
----------- | -----------|-----------|-----------
lambda_2 (Lambda)|(None, 27, 80, 3)|0|lambda_input_1[0][0]      
batchnormalization_12|(BatchNorm (None, 27, 80, 3)|54|batchnormalization_input_12[0][0]
(Convolution2D)|(None, 14, 40, 24)|1824|batchnormalization_12[0][0]      
(Convolution2D)|(None, 7, 20, 36)|21636|convolution2d_56[0][0] 
(Convolution2D)|(None, 7, 20, 48)|43248|convolution2d_57[0][0] 
(Convolution2D)|(None, 7, 20, 64)|27712|convolution2d_58[0][0] 
(Convolution2D)|(None, 7, 20, 64)|36928|convolution2d_59[0][0] 
(Flatten)|(None, 8960)|0|convolution2d_60[0][0]
(Dropout)|(None, 7, 20, 64)|0|convolution2d_85[0][0]          
(Dense)|(None, 1164)|10430604|flatten_12[0][0]                 
(Activation)|(None, 1164)|0|dropout_23[0][0]                 
(Dropout)|(None, 1164)|0|dense_56[0][0]                   
(Dense)|(None, 100)|116500|activation_45[0][0]                               
activation_46 (Activation)|(None, 100)|0|dropout_24[0][0]                 
(Dropout)|(None, 100)|0|dense_57[0][0]  
dense_58 (Dense)|(None, 50)|5050|activation_46[0][0]              
activation_47 (Activation)|(None, 50)|0|dense_58[0][0]                   
(Dropout)|(None, 100)|0|dense_57[0][0]  
dense_59 (Dense)|(None, 10)|510|activation_47[0][0]              
activation_48 (Activation)|(None, 10)|0|dense_59[0][0]                   
dense_60 (Dense)|(None, 1)|11|activation_48[0][0]              

Total params: 10684077

The learning rate we're choosing is 0.0001 according to other's discussion in on the Slack channel and our own testing. The 0.001 will be too large for training and 0.00001 will be to small to findout the optimized mininum.

### Conclusion
The model.json is with Lambda and Batch Normalization Layer while model3.json is only with Batch Normalization Layer. The previous one looks better.

We've also tested the data on the testing track, but it cannot generate good result. I believe it could be the reason that the neuroun network has remembered the training image data, with it's luminance, color and contrast info. I'll try to add some randomly contrast, color etc. to see if it can work better for testing track. 

However, the flipped image was proved to be deteriorate the training result

The .h5 file was too large to github.
