'''
Convolutional Neural Networks

helps to recognise images
2 px by 2 px images represented by a 2 by 2 2D array with pixel values of the colors
based on this information, computers are able to work with the images
this is for a black and white image

but for a colored image, we need to make a 3D array with R G and B for the third
dimension

Step 1: Convolution
Step 2: Max Pooling
Step 3: Flattening
Step 4: Full Connection

===Convolution===
use convolution function
there is a feature detector, usually a 3x3 matrix 
convolution is symbolised by a circle with a cross inside
feature detector is matched with EVERY part of the input image and see for resemblance
then it creates a feature map of the matches obtained
then we create many feature maps to obtain our first convolution layer
a convolution layer is just a compilation of feature maps
use several filters to detect certain features (emboss, edge detect, etc.)
convolution preserves spatial relationships between pixels

===ReLU (Rectified Linear Units)===
we're going to apply the rectifier function on the convolutional layer
rectifies function turns several values into simpler ones (e.g. black and white, grey turns to white)
this function normalizes the images that we're trying to analyse
rectifier function removes the negative values, leaving non-negative values

===Max Pooling, or Downsampling===
when images are squashed, or transformed, we need to make sure the network still recognises it
what about different types of images of the same object at different angles?
network must be "flexible" in this case, pooling takes care of this
max pooling picks regions from the feature map and picks the max value
why do we pool the features? when it's removing 75% of the information?
we're still preserving the features by taking the max pool algo
reducing size by 75% can be beneficial because we're both reducing size and introducing
spatial invariance, also helps processing, reducing number of parameters as well
preventing overfitting by limiting information that the model can fit into
subsampling is average pooling (averaging the numbers, instead of taking the max values)
Input IMage > Convolution > Pooling
features are PRESERVED in this step

===Flattening===
Flattening is mking the pooled feature map then put it into a long column
which can be processed by the ANN

===Full Connection===
adding an artificial neural network to the flattened data
in CNN, hidden layers are fully connected, in ANN it's not necessarily connected


'''

