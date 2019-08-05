import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pandas as pd
import time
import matplotlib.pyplot as plt



def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return  1.0/(1.0 + np.exp(-1.0*z))# your code here


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary
    
    train = []
    test = []
    train_data = np.zeros((50000,np.shape(mat["train0"])[1]))
    validation_data = np.zeros((10000,np.shape(mat["train0"])[1]))
    test_data = np.zeros((10000,np.shape(mat["test0"])[1]))
    train_label = np.zeros((50000,))
    validation_label = np.zeros((10000,))
    test_label = np.zeros((10000,))
    train_len=0
    validation_len=0
    test_len=0
    train_len_l=0
    validation_len_l=0
    for key in mat:
        if("train" in key):
            train = mat[key]#np.zeros(np.shape(mat.get(key)))
            shuffle = np.random.permutation(range(train.shape[0]))
            train_data[train_len:train_len+len(train)-1000] = train[shuffle[0:len(train)-1000],:]
            train_label[train_len_l:train_len_l+len(train)] = int(key[-1])
            validation_data[validation_len:validation_len+1000]=train[shuffle[len(train)-1000:len(train)],:]
            validation_len=validation_len+1000
            validation_label[validation_len_l:validation_len_l+1000] = int(key[-1])
            validation_len_l=validation_len_l+1000
            train_len=train_len+len(train)-1000
            train_len_l=train_len_l+len(train)-1000
        elif("test" in key):
            test = mat[key]#np.zeros(np.shape(mat.get(key)))
            test_data[test_len:test_len+len(test)] = test[np.random.permutation(range(test.shape[0])),:]
            test_label[test_len:test_len+len(test)] = key[-1]
            test_len=test_len+len(test)



    train_shuffle = np.random.permutation(range(train_data.shape[0]))
    train_data = train_data[train_shuffle]
    validate_shuffle = np.random.permutation(range(validation_data.shape[0]))
    validation_data = validation_data[validate_shuffle]
    test_shuffle = np.random.permutation(range(test_data.shape[0]))
    test_data = test_data[test_shuffle]
    train_data=(train_data)/255
    test_data=(test_data)/255
    validation_data=np.double(validation_data)/255
    train_label=train_label[train_shuffle]
    validation_label=validation_label[validate_shuffle]
    test_label = test_label[test_shuffle]

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
        
    # Feature selection
    # Your code here.
    data = np.array(np.vstack((train_data,validation_data,test_data)))
    
    features_count = np.shape(data)[1]

    #col_index = np.arange(features_count)

    count = np.all(data == data[0,:],axis=0)

    data = data[:,~count]

    train_data=data[0:len(train_data),:]
    validation_data=data[len(train_data):len(train_data)+len(validation_data),:]
    test_data=data[len(train_data)+len(validation_data):len(train_data)+len(validation_data)+len(test_data),:]
    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label

tracker = 0

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    training_data = np.concatenate((np.ones((np.shape(training_data)[0],1)),training_data),1)
    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    aj = np.dot(w1,np.transpose(training_data))
    zj=sigmoid(aj)
    zj = np.concatenate((np.ones((np.shape(np.transpose(zj))[0],1)),np.transpose(zj)),1)
    bj = np.dot(w2,np.transpose(zj))
    ol=sigmoid(bj)
    outputclass = np.zeros((n_class,training_label.shape[0])) #initialize all output class to 0
    i=0
    for i in range(len(training_label)):
        label=0
        label= int(training_label[i])
        outputclass[label,i] = 1
    error = -(np.sum((outputclass*np.log(ol)) + ((1-outputclass)*np.log(1-ol))))
    delta = ol - outputclass

    delta_new = (delta*(ol*(1-ol)))
    gradient_w2 = np.dot(delta_new,zj)
    product = ((1 - zj) * zj)*(np.dot(delta_new.T,w2))
    gradient_w1 = np.dot(np.transpose(product),training_data)


    error = error/np.shape(training_data)[0]


    sqr_weights = np.sum(w1**2) + np.sum(w2**2)

    grad = lambdaval * sqr_weights/(2*np.shape(training_data)[0])

    obj_val = error + grad

    gradient_w1 = (gradient_w1[1:n_hidden+1,] + (lambdaval*w1))/np.shape(training_data)[0]
    gradient_w2 = (gradient_w2 + lambdaval*w2)/np.shape(training_data)[0]
    obj_grad = np.concatenate((gradient_w1.flatten(),gradient_w2.flatten()),0)

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    #obj_grad = np.array([])

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here
    
#     dataWithBias = np.concatenate(( np.ones((data.shape[0], 1)),data),1)
    
#     hiddenLayerSigma = np.dot(dataWithBias, np.transpose(w1))
#     hiddenLayerOp = sigmoid(hiddenLayerSigma)
    
#     hiddenLayerOp = np.concatenate(( np.ones((hiddenLayerOp.shape[0],1)),hiddenLayerOp),1)
#     opLayerSigma = np.dot(hiddenLayerOp, np.transpose(w2))
#     finalOp = sigmoid(opLayerSigma)
    
#     global tracker
#     tracker = finalOp
#     labels = np.argmax(finalOp,axis=1)
    data = np.concatenate((np.ones((np.shape(data)[0],1)),data),1)
    aj = np.dot(w1,np.transpose(data))
    zj=sigmoid(aj)
    zj = np.concatenate((np.ones((np.shape(np.transpose(zj))[0],1)),np.transpose(zj)),1)
    bj = np.dot(w2,np.transpose(zj))
    ol=sigmoid(bj)
    labels = np.argmax(ol.T,axis=1)

    return labels


"""**************Neural Network Script Starts here********************************"""
colNames = ['Lambda','No_Hidden_Nodes','Train_Accuracy','Validation_Accuracy','Test_Accuracy','Time']
obs = pd.DataFrame(columns = colNames)

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

noOfHiddenNodes = np.arange(0,100,5)
#lambas = np.arange(0,60,10)

#noOfHiddenNodes = [50]
lambas = [0]

count = 0
#  Train Neural Network
for l in lambas:
    for hns in noOfHiddenNodes:
# set the number of nodes in input unit (not including bias unit)
        
        startTime = time.time()
        
        n_input = train_data.shape[1]
        
        # set the number of nodes in hidden unit (not including bias unit)
        n_hidden = hns
        
        # set the number of nodes in output unit
        n_class = 10
        
        # initialize the weights into some random matrices
        initial_w1 = initializeWeights(n_input, n_hidden)
        initial_w2 = initializeWeights(n_hidden, n_class)
        
        # unroll 2 weight matrices into single column vector
        initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)
        
        # set the regularization hyper-parameter
        lambdaval = l
        
        args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)
        
        # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
        
        opts = {'maxiter': 50}  # Preferred value.
        
        nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
        
        # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
        # and nnObjGradient. Check documentation for this function before you proceed.
        # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)
        
        
        # Reshape nnParams from 1D vector into w1 and w2 matrices
        w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
        w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
        
        # Test the computed parameters
        
        predicted_label_train = nnPredict(w1, w2, train_data)
        
        # find the accuracy on Training Dataset
        print('\n For Number of Nodes and Lambda ' +str(hns) +'and '+str(l))
        
        print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_train == train_label).astype(float))) + '%')
        train_acc = str(100 * np.mean((predicted_label_train == train_label).astype(float)))
        predicted_label_validate = nnPredict(w1, w2, validation_data)
        
        # find the accuracy on Validation Dataset
        
        print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_validate == validation_label).astype(float))) + '%')
        val_acc = str(100 * np.mean((predicted_label_validate == validation_label).astype(float)))
        predicted_label_test = nnPredict(w1, w2, test_data)
        
        # find the accuracy on Validation Dataset
        
        print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_test == test_label).astype(float))) + '%')
        #collections.Counter(predicted_label)
        test_acc = str(100 * np.mean((predicted_label_test == test_label).astype(float)))
        endTime = time.time()
        runtime = endTime-startTime
        
        obs.loc[count] = [l,hns,train_acc,val_acc,test_acc,runtime]
        
        count += 1