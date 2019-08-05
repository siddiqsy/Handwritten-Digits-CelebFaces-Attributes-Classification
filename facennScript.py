'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return  1.0/(1.0 + np.exp(-1.0*z))# your code here
    
    
# Replace this with your nnObjFunction implementation
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
    
# Replace this with your nnPredict implementation
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
    data = np.concatenate((np.ones((np.shape(data)[0],1)),data),1)
    aj = np.dot(w1,np.transpose(data))
    zj=sigmoid(aj)
    zj = np.concatenate((np.ones((np.shape(np.transpose(zj))[0],1)),np.transpose(zj)),1)
    bj = np.dot(w2,np.transpose(zj))
    ol=sigmoid(bj)
    labels = np.argmax(ol.T,axis=1)

    return labels

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network

colNames = ['Lambda','No_Hidden_Nodes','Train_Accuracy','Validation_Accuracy','Test_Accuracy','Time']
obs_facenn = pd.DataFrame(columns = colNames)

noOfHiddenNodes = np.arange(0,300,25)
lambas = np.arange(0,20,5)

#noOfHiddenNodes = [50]
#no_of_classes = [2]

count = 0
#  Train Neural Network
for l in lambas:
    for hns in noOfHiddenNodes:
        startTime = time.time()
        # set the number of nodes in input unit (not including bias unit)
        n_input = train_data.shape[1]
        # set the number of nodes in hidden unit (not including bias unit)
        n_hidden = hns
        # set the number of nodes in output unit
        n_class = 2

        # initialize the weights into some random matrices
        initial_w1 = initializeWeights(n_input, n_hidden);
        initial_w2 = initializeWeights(n_hidden, n_class);
        # unroll 2 weight matrices into single column vector
        initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
        # set the regularization hyper-parameter
        lambdaval = l;
        args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

        #Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
        opts = {'maxiter' :50}    # Preferred value.

        nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
        params = nn_params.get('x')
        #Reshape nnParams from 1D vector into w1 and w2 matrices
        w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
        w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

        # Test the computed parameters

        predicted_label_train = nnPredict(w1, w2, train_data)

        # find the accuracy on Training Dataset
        print('\n For Number of Nodes and no of class ' +str(hns) +'and '+str(nc))

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

        obs_facenn.loc[count] = [l,hns,train_acc,val_acc,test_acc,runtime]

        count += 1