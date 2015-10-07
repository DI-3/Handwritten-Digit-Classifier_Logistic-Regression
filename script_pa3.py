import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle
from sklearn.svm import SVC

def preprocess():
    """ 
     Input:
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
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection
    """
    
    mat = loadmat('mnist_all.mat'); #loads the MAT object as a Dictionary
    
    n_feature = mat.get("train1").shape[1];
    n_sample = 0;
    for i in range(10):
        n_sample = n_sample + mat.get("train"+str(i)).shape[0];
    n_validation = 1000;
    n_train = n_sample - 10*n_validation;
    
    # Construct validation data
    validation_data = np.zeros((10*n_validation,n_feature));
    for i in range(10):
        validation_data[i*n_validation:(i+1)*n_validation,:] = mat.get("train"+str(i))[0:n_validation,:];
        
    # Construct validation label
    validation_label = np.ones((10*n_validation,1));
    for i in range(10):
        validation_label[i*n_validation:(i+1)*n_validation,:] = i*np.ones((n_validation,1));
    
    # Construct training data and label
    train_data = np.zeros((n_train,n_feature));
    train_label = np.zeros((n_train,1));
    temp = 0;
    for i in range(10):
        size_i = mat.get("train"+str(i)).shape[0];
        train_data[temp:temp+size_i-n_validation,:] = mat.get("train"+str(i))[n_validation:size_i,:];
        train_label[temp:temp+size_i-n_validation,:] = i*np.ones((size_i-n_validation,1));
        temp = temp+size_i-n_validation;
        
    # Construct test data and label
    n_test = 0;
    for i in range(10):
        n_test = n_test + mat.get("test"+str(i)).shape[0];
    test_data = np.zeros((n_test,n_feature));
    test_label = np.zeros((n_test,1));
    temp = 0;
    for i in range(10):
        size_i = mat.get("test"+str(i)).shape[0];
        test_data[temp:temp+size_i,:] = mat.get("test"+str(i));
        test_label[temp:temp+size_i,:] = i*np.ones((size_i,1));
        temp = temp + size_i;
    
    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis = 0);
    index = np.array([]);
    for i in range(n_feature):
        if(sigma[i] > 0.001):
            index = np.append(index, [i]);
    train_data = train_data[:,index.astype(int)];
    validation_data = validation_data[:,index.astype(int)];
    test_data = test_data[:,index.astype(int)];

    # Scale data to 0 and 1
    train_data = train_data/255.0;
    validation_data = validation_data/255.0;
    test_data = test_data/255.0;
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z));
    
def blrObjFunction(params, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args
    
    n_data = train_data.shape[0];
    n_feature = train_data.shape[1];
    error = 0;
    error_grad = np.zeros((n_feature+1,1));
    
    ##################
    # YOUR CODE HERE #
    ##################
    
    # Step-1: Compute yn
    w = params.reshape((n_feature+1, 1))
    wt = w.transpose();
    xn = train_data;
    xn_with_bias = np.array([])
    xn_with_bias_list = [] 
    for x in range (0, n_data):
        xn_with_bias_list.append(np.append ([1], xn[x]))
    xn_with_bias = np.vstack(xn_with_bias_list)
    yn = sigmoid(np.dot (xn_with_bias, w));
    
    # Step-2: Compute Error
    tn = labeli;
    sum1 = tn * np.log (yn);
    sum2 = (1-tn) * np.log ((1- yn));
    errorr = 0
    for i in range (0, n_data):
        errorr = errorr + sum1[i] + sum2[i];
    errorr = -1 * errorr;
    error = errorr[0]   
     
    # Step-3: Compute Error gradient
    yn_minus_tn = yn - tn;
    xn_with_bias
    error_grad = np.dot(xn_with_bias.transpose(), yn_minus_tn);
    error_grad = np.squeeze(np.asarray(error_grad));
    print (error)
    return error, error_grad

def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """

    #Step 1: Add bias to data. Bias is to be added in the beginning
    data_with_bias = np.array([])
    data_with_bias_list = []
    for x in range (0, data.shape[0]):
        #TODO:Check if value 1 can be appended or not
        data_with_bias_list.append(np.append ([1], data[x]))
    data_with_bias = np.vstack(data_with_bias_list)
    
    prod = np.dot (data_with_bias, W);
    yn = sigmoid(prod)
    
    label_list = []
    for i in range(0, yn.shape[0]):
        label_list.append(np.argmax(yn[i]))

    label = np.array([])
    label = np.vstack(label_list)
    
    return label



"""
Script for Logistic Regression
"""
train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();

# number of classes
n_class = 10;

# number of training samples
n_train = train_data.shape[0];

# number of features
n_feature = train_data.shape[1];

T = np.zeros((n_train, n_class));
for i in range(n_class):
    T[:,i] = (train_label == i).astype(int).ravel();
    
# Logistic Regression with Gradient Descent
W = np.zeros((n_feature+1, n_class));
initialWeights = np.zeros((n_feature+1,1));
opts = {'maxiter' : 50};
for i in range(n_class):
    labeli = T[:,i].reshape(n_train,1);
    args = (train_data, labeli);
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
    W[:,i] = nn_params.x.reshape((n_feature+1,));
print('\n Finished Executing the OBJ fn.');
# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data);
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data);
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data);
print('\n Testing set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

f = open('params.pickle','wb')
pickle.dump(W,f)
f.close()

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
################  Linear Model ########################
clf = SVC(kernel = 'linear')
clf.fit(train_data, train_label.ravel())
print('Linear Model - 1')
print ('\n Training set Accuracy:  '+str(clf.score(train_data,train_label.ravel())* 100) + '%')
print ('\n Validation set Accuracy:'+ str(clf.score(validation_data,validation_label.ravel()) * 100) + '%')
print('\n Testing set Accuracy:    ' + str(clf.score(test_data,test_label.ravel()) * 100) + '%')


#
###############  RBF Gamma - 1 #######################
#
print('RBF GAMMA - 1')
clf = SVC(kernel = 'rbf',gamma = 1.0 )
clf.fit(train_data, train_label.ravel())
print ('\n Training set Accuracy:  '+str(clf.score(train_data,train_label.ravel()) * 100) + '%')
print ('\n Validation set Accuracy:'+ str(clf.score(validation_data,validation_label.ravel()) * 100) + '%')
print('\n Testing set Accuracy:    ' + str(clf.score(test_data,test_label.ravel()) * 100) + '%')

#
###############  RBF Gamma - 0  #######################
#

print('RBF GAMMA - 0- Default')
clf = SVC(kernel = 'rbf')
clf.fit(train_data, train_label.ravel())
print ('\n Training set Accuracy:  '+str(clf.score(train_data,train_label.ravel()) * 100) + '%')
print ('\n Validation set Accuracy:'+ str(clf.score(validation_data,validation_label.ravel()) * 100) + '%')
print('\n Testing set Accuracy:    ' + str(clf.score(test_data,test_label.ravel()) * 100) + '%')


#
##########  RBF  GAmma default  for varying values of C #######################
#
print('##########  RBF  GAmma default  1 to 100 ######################')
for i in np.array([1,10,20,30,40,50,60,70,80,90,100]) :
    
    print ('RBF GAMMA - 0 ,C - The run number')
    print (i)
    clf = SVC(kernel = 'rbf', C = i)
    clf.fit(train_data, train_label.ravel())
   
    print ('\n Training set Accuracy:  '+str(clf.score(train_data,train_label.ravel()) * 100) + '%')
    print ('\n Validation set Accuracy:'+ str(clf.score(validation_data,validation_label.ravel()) * 100) + '%')
    print('\n Testing set Accuracy:    ' + str(clf.score(test_data,test_label.ravel()) * 100) + '%')




