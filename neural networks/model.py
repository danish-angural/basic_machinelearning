import numpy as np
import matplotlib.pyplot as plt
import h5py
from nn_utils import *
np.random.seed(1)

def load_data():
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes



def gradient_descent(X, Y, layers_dims, learning_rate, num_iterations, lambd):
    costs=[]
    print_cost=True
    parameters=initialize(layers_dims)

    # s, v= initialize_adam(parameters)
    t=0
    plt.plot(costs)
    for i in range(0, num_iterations):
        t=t+1
        AL, caches =forward(X, parameters, mode="train")
        cost=compute_cost(AL, Y, parameters, lambd)
        grads=backward(AL, Y, caches, lambd)
        parameters= update_parameters(parameters,grads, learning_rate)
        # parameters, v, s=update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
        if print_cost and i%100==0:
            print ("Cost after iteration %i: %f" %(i, cost))
        costs.append(cost)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

train_x = train_x_flatten/255.
test_x = test_x_flatten/255.
layers_dims = [12288, 20, 7, 5, 1]

# parameters = gradient_descent(train_x, train_y, layers_dims, learning_rate=0.075, num_iterations = 2500, lambd=10)
# pred_train = predict(train_x, train_y, parameters, "train")
# pred_test = predict(test_x, test_y, parameters, "test")


def softmax(z):
    return z/(np.sum(z, axis=1, keepdims=True))

z=[[1, 1, 1], [0, 0, 0], [0, 1, 0]]
print(softmax(z))
