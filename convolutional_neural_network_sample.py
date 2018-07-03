# CNN

# this is a sample for classifying only two types of images cats and dogs
# you can replcae the parameters and the images with other images to develop
# another CNN that will suit your preferences
# e.g. brain tumor/medical imaging

# we need to do some image preprocessing to do that first

# images must be organised in such a way such that they're organised in folders
# of test and training sets, then name the images by their category (e.g. cat_01, dog_01)
# so testset > test/traiing > cat/dog
# we can extract the names and get the dependent variable vector
# but we won't do that, we'll use keras instead to import images in an efficient way
# prepare a special structure or the data set
# see the structure of the dataset folder

# the independent variable now is a 3D array

# First we need to build the Convolutional Neural Network
# the data preprocessing is done by organising the folder
from keras.models import Sequential

# Convolution2D is to add the convolution layers, images is 2D
from keras.layers import Conv2D

# this package is to proceed to the pooling step that will add the pooling layers
from keras.layers import MaxPooling2D

# then we need to flatten all the feature maps into the large feature vector
from keras.layers import Flatten

# dense is to add the fully connected layers in a classic ANN
from keras.layers import Dense

# then we need to initialize the CNN
# create the classifier
classifier = Sequential()

# add first layer
# convolution step is here
# convolution is when pictures are converted into smaller n by n pixels 
# then we use the feature detector to create a feature map
# last time we used Dense but now we need to use Convolution2D
# filters is the number of feature maps that we want to create
# convolution kernel is the feature detector
# col and row is the feature detector dimensions
# so 64, 3, 3 means that we will create 64 feature detectors with 3 by 3 feature detectors
# it's common practice to use 32 feature detectors
# input_shape is the shape of the input image that we're going to which we're going to 
# apply our feature detectors, important as images we have not same size and format
# we need to force them to beome same format
# thus we need to specify which expectedformat this images are going to be converted to
# since we're using colored images, we are going to use a 3D array
# thus here input_shape is expecting colored images with 256 by 256 pixel images
# since we're using tensorflow backend, we need to order input_shape differently
# then we need to specify an activation function, use rectifier function to remove
# negative pixels, making sure the non-linearity of the model
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = "relu"))


# now we have finished the convolution step, we need to pool the feature map
# feature map is then converted into a pooled feature map, sleecting the highest
# value of the feature map for every square pixel size we choose
# then we compile everything into a pooling layer
# this is so that we can get a compact flattened vector to push into the ANN
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# add another convol layer to imporve accuracy of test set
# input is not going to be images, it's from the pooled layer
# no need to put input_shape, only need to put in the first one because keeras
# will know and use the previous one
# you can also add twice the amount of feature detectors for every convol layer
# you add, so for example 64 on the third one, it's common practivce to double it
classifier.add(Conv2D(32, (3, 3), activation = "relu"))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

# then we need to flatten everything into one vector
# why don't we lose the spatial structure by flattening it into a vector?
# by creating our feature maps, we already extracted the features, the high numbers 
# in the feature maps represent the features in the images themselves
# in this vector we still keep the features in the image
# if we immediately flatten the image to a vector, we don't get any information
# of the features of the image itself (think of patterns)
# basically we will not get any information regarding the features of the image
# if we don't convolute and pool it first
classifier.add(Flatten())
# no need parameters because keras understands

# here is the full connection step, making a classic ANN with fully connected layers
# we need to use the flattened vector as the input layer to classify them
# we create the hidden layers from here on
# we have tons of input nodes, but we don't really know the amount, we shouldn't take a small number
# choice of number comes from experience, not too low not too high
# common practice to put a power of 2
classifier.add(Dense(units = 128, activation = 'relu'))

# this is the output layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# we need to compile the CNN now
# choose optimizer to choose the stochastic gradient descent algorithm
# since we have binary outcome we use the logarithmic loss function (binary cross_entropy)
# if we have >2 then we choose categorical_crossentropy
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# before we get our result, we need to fit our images to the CNN, leggoo!!
# use Keras for image augmentation, preprocessing images to prevent overfitting
# when great results on training set but nah on test set
# overfitting happens when there is not enough observations in the training set
# it is actually not much if we use 10,000, then we need to use data augmentation
# it will create random selection/batches of images, shearing, rotating, etc to transform
# a lot of images are augmented, allowing us to enrich our data set without large image base
# refer to https://keras.io/preprocessing/image/ for the template

from keras.preprocessing.image import ImageDataGenerator

# this is the image augmentation part, where we shear, transform etc. etc.
# rescale makes pixel values between 0 and 1
# the range is the extent whether we want to apply the shear/zoom
train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)

# for test set we will onlt rescale the pixels of the images
test_datagen = ImageDataGenerator(rescale=1./255)

# this is where we create the training set
# target size is the size that is expected from the ANN same as the layers above
# batch_size is the number of batches that our images are going to be included
# after which weights will be updated
# class_mode is when independent variable is binary or not
# if we increase the size that the image is going to be resized
# we will have more information to take, increase the target_size
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
# this code will detect the 2 categories

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# here we will fit the ANN to the sets
# samples per epoch is the number of images we have in the training set
# all the images passed each epoch is 8000
# validation_steps is the number of images in our test set
classifier.fit_generator(
        training_set,
        steps_per_epoch = 8000,
        epochs = 25,
        validation_data = test_set,
        validation_steps = 2000)
# we want a smaller difference in accuracy for test and training set

# we now need to improve the accuracy of the test set, then we need to increase the depth of the CNN
# add another convolutional layer
# add fully connected layer (2 options)
# better option is to add another convolutional layer

# now we make a new single prediction using CNN
# first we need to preprocess the image by numpy
import numpy as np
from keras.preprocessing import image

# load our image where we want to make our prediction
# target_size MUST be equal to the target_size in the training_set
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))

# now we have to use another function to transform the image to the 3D array same format as the image
# in the input layer
# turn into (64, 64) by 3
test_image = image.img_to_array(test_image)

# but prediction is supposed to have 4 dimensions in .predict
# this new dimension corresponds to the batch the image is in
# in general the predict function cannot accept an input layer individually
# must be inside a batch where it contains the input
# the axis in expand_dims is the position of the index where the dimension is in
# this batch is in the first index
test_image = np.expand_dims(test_image, axis=0)

# then we need to make 1 dimension 
result = classifier.predict(test_image)

# we will then see which one is which
training_set.class_indices

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

