import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#############################################################################
def softmax(z, size):
    z_sub = np.array(z)
    for i in range(size):
        z_max = np.argmax(z[:,i])
        z_sub[:,i] = z_sub[:,i]-z_max
        y = np.exp(z_sub[:,i])
        y_sum = np.sum(y)
        z_sub[:,i] = y/y_sum
        
    return z_sub

def accuracy(Y, t, size):
    error = np.zeros(10)
    for i in range(size):
        j = np.argmax(Y[:, i])
        label = signal.unit_impulse(10, j)
        if np.dot(label, t[i,:]) != 1:
            j = np.argmax(t[i,:])
            error[j] = error[j] + 1
    
    accuracy = np.average(1 - error/size)
    return accuracy

def risk(Y, t, size):
    ln_Y = np.log(Y)
    ln_Y = np.multiply(ln_Y, t.T)
    losses = -1 * np.sum(ln_Y, axis=0)
    risk = np.sum(losses)
    risk = risk/size
    return risk

def grad(Y, t, X, dataset_size, data_size):
    delta = t.T - Y
    grad = np.zeros((10,data_size))
    for i in range(dataset_size):
        for j in range(10):
            grad[j, :] = grad[j, :] + delta[j, i]*X[i,:]
    
    return grad

###############################################################################
train_size = np.shape(mnist.train.images)
test_size = np.shape(mnist.test.images)
val_size = np.shape(mnist.validation.images)

# Normalize Data
for i in range(train_size[0]):
    mnist.train.images[i,:] = (mnist.train.images[i,:]) * 2 - 1
    if i < test_size[0]:
        mnist.test.images[i,:] = (mnist.test.images[i,:]) * 2 - 1
    if i < val_size[0]:
        mnist.validation.images[i,:] = (mnist.validation.images[i,:]) * 2 - 1
        
image = mnist.train.images[5,:]
image = image.reshape((28,28))

plt.imshow(image, cmap='gray')
plt.show()

###############################################################################
# Weight Initialization
sigma = 10**-3
W = sigma*np.random.randn(10, train_size[1])

learn_null = 10**-5
learn_rate = learn_null
T = 32.0

epochs = 1024

#Training
accuracies = np.zeros((epochs, 2))
risks = np.zeros((epochs, 2))
best_W = W

for i in range(epochs):
    
    Z_train = np.matmul(W, mnist.train.images.T)
    Z_val = np.matmul(W, mnist.validation.images.T)
    Y_train = softmax(Z_train, train_size[0])
    Y_val = softmax(Z_val, val_size[0])
    accuracies[i, 0] = accuracy(Y_train, mnist.train.labels, train_size[0])
    accuracies[i, 1] = accuracy(Y_val, mnist.validation.labels, val_size[0])
    risks[i, 0] = risk(Y_train, mnist.train.labels, train_size[0])
    risks[i, 1] = risk(Y_val, mnist.validation.labels, val_size[0])
    gradient = grad(Y_train, mnist.train.labels, mnist.train.images, train_size[0], train_size[1])
    
    if i > 0:
        if accuracies[i, 1] > accuracies[i-1, 1]:
            best_W = W
            
    W = W - learn_rate * gradient
    
    learn_rate = learn_null / (1 + i/T)
    print("Accuracy: {0} {1}".format(str(accuracies[i,0]), str(accuracies[i,1])))
    
    
plt.plot(range(epochs), accuracies[:, 0], range(epochs), accuracies[:, 1])
plt.show()
