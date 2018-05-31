# **Traffic Sign Recognition** 

## Writeup For Leon Lu

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! This is my project repo

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32*32*3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Show one image for every image class.


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I chose to normalize the image set but not converted them into grayscale. Since i want to see if the network could figure out how (un)important the color is.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3x50     	| 1x1 stride 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride		|
| Convolution 5x5x40    | 1x1 stride      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride		|
| Fully connected		| etc.        									|
| Fully connected		| etc.        									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used :
loss : tf.reduce_mean(softmax_cross_entropy_with_logits)
optimizer : AdamOptimizer
batch size : 128
number of epochs : 20
learning rate : 0.001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.74%
* validation set accuracy of 94.3% 
* test set accuracy of 93.05%

For the model architechture, I went directly for the LeNet since it's known for handling images. The architeture starts with two convolutional layers with relu and max pooling in between. Convolutional layers are good at taking advantage of the spatial relationship of pixels in an image and aggregate those information into higher level representation of the features. Later with the help of fully connected layers, we could classify the images basing on higher level features instead of directly looking at the pixels.

At first the network size was too small to capture the essence of the image data set, so I make the feature maps of the first 2 conv layer deeper and the fully connected layers bigger. With a bigger network the accuracy seems to go up with more iterations, therefore I tune up the number of epochs to make sure it has enough iterations to learn from the data.

Final model is overfitting to the dataset. Potential improvements could be made by adding dropouts, make the network smaller, converting images to grayscale etc.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I chose several images randomly from the web, most of them are not in the database. I'm intersted to see how close the network could get. The images looks different but not that different from the database.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| 100 km   									| 
| 30 km/h     			| 100 km										|
| No passing					| 100 km											|
| No U-turn	      		| ?					 				|
| Dead end			| 100 km      							|


The model was only able to classify 1 image correctly, which is somewhat expected since it's overfitting to the training dataset. Also other images are classified pretty badly, they don't look similar to the answer it gives.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The model seems to be very certain of it's decision although they are wrong.


