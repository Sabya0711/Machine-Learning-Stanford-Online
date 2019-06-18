
##Coursera -ML
##Exercise -3
##python version 3.6
##directory structure--> code= '/coursera/scripts' ; data = '/coursera/data'

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#for importing octave matrix files (.mat)
import scipy.io
#for showing number matrix in a image format
import scipy.misc
#for specific colormap for images 
import matplotlib.cm as cm
#for display random image
import random
#for vectorized sigmoid function implementation
from scipy.special import expit


## MULTI-CLASS CLASSIFICATION
## DATASET 1.1

filename = 'data/ex3data1.mat'
datamat = scipy.io.loadmat(filename)
X,y = datamat['X'], datamat['y']
## adding vector of 1's X matrix 
X = np.insert(X,0,1,axis=1)
print("Shape of X matrix", X.shape)
print("Shape of y matix", y.shape)
print("Unique numbers in y", np.unique(y))
print("Shape of X[0]", X[0].shape)


## VISUALIZING THE DATA
def getDatumImg(row):
    """
    Function that is handed a single np array with shape 1x400,
    crates an image object from it, and returns it
    """
    width, height = 20, 20
    square = row[1:].reshape(width,height)
    return square.T

def displayData(indices_to_display = None):
    """
    Function that picks 100 random rows from X, creates a 20x20 image from each,
    then stitches them together into a 10x10 grid of images, and shows it.
    """
    width, height = 20, 20
    nrows, ncols = 10, 10
    if not indices_to_display:
        indices_to_display = random.sample(range(X.shape[0]), nrows*ncols)
        
    big_picture = np.zeros((height*nrows,width*ncols))
    
    irow, icol = 0, 0
    for idx in indices_to_display:
        if icol == ncols:
            irow += 1
            icol  = 0
        iimg = getDatumImg(X[idx])
        big_picture[irow*height:irow*height+iimg.shape[0],icol*width:icol*width+iimg.shape[1]] = iimg
        icol += 1
    fig = plt.figure(figsize=(6,6))
    img = scipy.misc.toimage( big_picture )
    plt.imshow(img,cmap = cm.Greys_r)


displayData()


## VECTORIZING LOGISTIC REGRESSION

## Hypothesis function and cost function for logistic regression 
def h(mytheta,myX): #Logistic hypothesis function
    return expit(np.dot(myX,mytheta))

## simpler cost function
def computeCost(mytheta,myX,myy,mylambda = 0.):
    m = myX.shape[0] #5000
    myh = h(mytheta,myX) #shape: (5000,1)
    term1 = np.log( myh ).dot( -myy.T ) #shape: (5000,5000)
    term2 = np.log( 1.0 - myh ).dot( 1 - myy.T ) #shape: (5000,5000)
    left_hand = (term1 - term2) / m #shape: (5000,5000)
    right_hand = mytheta.T.dot( mytheta ) * mylambda / (2*m) #shape: (1,1)
    return left_hand + right_hand #shape: (5000,5000)

## ONE-VS-ALL CLASSIFICATION 

## Using python's fmin_cg to optimize the cost function using conjugate gradient algorithm

def costGradient(mytheta,myX,myy,mylambda = 0.):
    m = myX.shape[0]
    beta = h(mytheta,myX)-myy.T 
    
    #regularization skips the first element in theta
    reg = mytheta[1:]*(mylambda/m) 
    gradient = (1./m)*np.dot(myX.T,beta) 
    
    #regularization skips the first element in theta
    gradient[1:] = gradient[1:] + reg
    return gradient

from scipy import optimize

def optimizeTheta(mytheta,myX,myy,mylambda=0.):
    myresult = optimize.fmin_cg(computeCost, fprime=costGradient, x0=mytheta, \
                              args=(myX, myy, mylambda), maxiter=50, disp=False,\
                              full_output=True)
    return myresult[0], myresult[1]

def buildTheta():
    """
    Function that determines an optimized theta for each class
    and returns a Theta function where each row corresponds
    to the learned logistic regression params for one class
    """
    mylambda = 0.
    initial_o = np.zeros((X.shape[1],1)).reshape(-1)
    Theta = np.zeros((10,X.shape[1]))
    for i in xrange(10):
        iclass = i if i else 10 #class "10" is handwritten zero
        print ("Optimizing for handwritten number %d..."%i)
        logic_Y = np.array([1 if x == iclass else 0 for x in y])#.reshape((X.shape[0],1))
        itheta, imincost = optimizeTheta(initial_o,X,logic_Y,mylambda)
        Theta[i,:] = itheta
    print ("Done!")
    return Theta

theta_output  = buildTheta()

def predictOneVsAll(myTheta,myrow):
    """
    Function that computes a hypothesis for an individual image (row in X)
    and returns the predicted integer corresponding to the handwritten image
    """
    classes = [10] + range(1,10)
    hypots  = [0]*len(classes)
    #Compute a hypothesis for each possible outcome
    #Choose the maximum hypothesis to find result
    for i in xrange(len(classes)):
        hypots[i] = h(myTheta[i],myrow)
    return classes[np.argmax(np.array(hypots))]

## Calculating training set accuracy

correct_pred, n_total = 0., 0.
incorrect_indices = []
for irow in xrange(X.shape[0]):
    n_total += 1
    if predictOneVsAll(Theta,X[irow]) == y[irow]: 
        correct_pred += 1
    else: incorrect_indices.append(irow)
print ("Training set accuracy: %0.1f%%"%(100*(correct_pred/n_total)))


#Displaying the incorrecly predicted indices
displayData(incorrect_indices[:100])
displayData(incorrect_indices[100:200])
displayData(incorrect_indices[200:300])


## NEURAL NETWORKS

## loading the pre-trained weights given 
datafile = 'data/ex3weights.mat'
datamat = scipy.io.loadmat(datafile)
Theta1, Theta2 = mat['Theta1'], mat['Theta2']
print ("Shape of Theta1:",Theta1.shape)
print ("Shape of Theta2:",Theta2.shape)

##FEEDFORWARD PROPOGATION

print("FEEDFORWARD PROPOGATION")

def forwardProp(row,Thetas):
    """
    Function that given a list of Thetas, propagates the
    Row of features forwards, assuming the features already
    include the bias unit in the input layer, and the 
    Thetas need the bias unit added to features between each layer
    """
    features = row
    for i in xrange(len(Thetas)):
        Theta = Thetas[i]
        z = Theta.dot(features)
        a = expit(z)
        if i == len(Thetas)-1:
            return a
        a = np.insert(a,0,1) #Add the bias unit
        features = a


def predict_NN(row,Thetas):
    """
    Function that takes a row of features, propagates them through the
    NN, and returns the predicted integer that was hand written
    """
    classes = range(1,10) + [10]
    output = forwardProp(row,Thetas)
    return classes[np.argmax(np.array(output))]

##calculating the accuracy with NN

print("Accuracy with forprop...")

myThetas = [ Theta1, Theta2 ]
correct_pred, n_total = 0., 0.
incorrect_indices = []
for irow in xrange(X.shape[0]):
    n_total += 1
    if predict_NN(Theta,myThetas) == int(y[irow]): 
        correct_pred += 1
    else: incorrect_indices.append(irow)
print ("Training set accuracy with NN : %0.1f%%"%(100*(correct_pred/n_total)))

print("DISPLAYING THE IMAGES INCORRECTLY PREDICTED")


for x in xrange(5):
    i = random.choice(incorrect_indices)
    fig = plt.figure(figsize=(3,3))
    img = scipy.misc.toimage( getDatumImg(X[i]) )
    plt.imshow(img,cmap = cm.Greys_r)
    predicted_val = predict_NN(X[i],myThetas)
    predicted_val = 0 if predicted_val == 10 else predicted_val
    fig.suptitle('Predicted: %d'%predicted_val, fontsize=12, fontweight='bold')


print("IMO, the images which are predicted incorrectly as some other image has some shared features with the \
	class they being classified as. For instance, one of the image displayed as '2' has been predicted as '0'. This is clearly because \
	the '2' it has been written has a curve on the right edge which resembles right curve of '0', and hence the prediction. \
	Better training is required to capture these subtle features. ")
############### END OF SCRIPT ################












































