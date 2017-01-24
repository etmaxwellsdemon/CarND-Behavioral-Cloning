# Self-Driving Car Engineer Nanodegree
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

1. At first, the dataset was created by keyboard(arrow keys). However, because of the limit of the keyboard, the training data is not smooth enough. For my own data set, I have 3 laps of clockwise, 3 lap of counter-clockwise, 1 lap of recovery from left half and 1 lap of recovery from right. 

2. Then I tried the dataset created by [Udacity officially](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)

3. I've also tried more datasets generated by Map2 or add additional training datas/create mirror images, but the result sometimes turns out to be overfitted.

### Preprocessing

![](https://github.com/etmaxwellsdemon/CarND-Behavioral-Cloning/blob/master/YUV.png)

1. We convert the RGB to YUV according to the [paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

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

#####Network Structure Parameter
Other parameters of network was tested. I've tried to divide conv2 layer's output by 2, which turns out the result for conv2d to: 

Layer (type)|Output Shape                 
----------- | -----------
(Convolution2D)|(None, 14, 40, 12)     
(Convolution2D)|(None, 7, 20, 18)
(Convolution2D)|(None, 7, 20, 23)
(Convolution2D)|(None, 7, 20, 32)
(Convolution2D)|(None, 7, 20, 32)

But the result doesn't change much.

I've also tested the network structure with and without Lambda layer, and it turns out the with the Lambda layer one runs more smoothly.

#####Learning Rate
The learning rate we're choosing is 0.0001 according to other's discussion in on the Slack channel and our own testing. The 0.001 will be too large for training and 0.00001 will be to small to findout the optimized mininum.


#####Epoch
What's more, the reason of choosing 20 as the epoch parameter is that, I've found usually 10+ of epoches will sufficient to reach a good val_loss. And I've added the patience parameter of 2, which will make the training stop if the loss result for validation data stop decreasing for more than twice.

#####Batch Size
128 and 64 was tested, it turns out that 128 was a good value for batch size, there's not much theory and experiment behind it.



### Conclusion

With the official training data provided by Udacity, the car was able to run smoothly by itself on testing data of Track1.

However, with the weights training on Track1, the car cannot generate good result on Track2. I believe it could be the reason that the neuroun network has remembered the training image data, with it's luminance, color and contrast info. 

So I've also tried to add Track2 data with Track1 for training in case of overfitting. The result is much more better, but still hit the track after a while on Track2.

However, the result still runs smoothly on Track1.

Other ways to improve the result could be adding some randomly contrast, color etc. and I'll try them later.

In addition, the throttle value in drive.py also effects the result. I guess it could be the reason that under small turning radius, a large throttle will make the car hard to control, in other words, overshoot the control. So the I set throttle to 0.1 to make sure of the result.

As the .h5 file was too large to github, [I've uploaded it on to  Dropbox](https://www.dropbox.com/s/jneba7kdg0p2oim/model.h5?dl=0). 

In case of missing the result, I've uploaded it on [YouTube](https://youtu.be/k1dibDf4xx0)

