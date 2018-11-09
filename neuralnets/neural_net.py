import numpy as np
from scipy import signal
from __future__ import division
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#########################################
# Pre-Processing Steps
#########################################

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

train_images = mnist.train.images
train_labels = mnist.train.labels
val_images = mnist.validation.images
val_labels = mnist.validation.labels
test_images = mnist.test.images
test_labels = mnist.test.labels

# image = mnist.train.images[5,:]
# image = image.reshape((28,28))

########################################
# Variable Initialization
########################################

# Layer Weights
mu = 0
sigma = 10**-3
w_1 = np.random.normal(mu, sigma, (64, 785))
w_2 = np.random.normal(mu, sigma, (10, 65))

# Learning Rate
learn_rate = 10**-2
base_rate = 10**-2
T = 8

# Numerical Variables
epsilon = np.arange(10**-3, 1.06 * 10**-1, 5 * 10**-3)
ep_shape = np.shape(epsilon)
n = np.arange(0, 4, 2)
errors = np.zeros((ep_shape[0], 4 * len(n)))

# Training Variables
batch = 128
epochs = 64
train_loss = []
train_accuracy = []
accuracy_epoch = np.zeros((64, 2))
risk_epoch = np.zeros((64, 2))
best_w_1 = np.zeros(np.shape(w_1))
best_w_2 = np.zeros(np.shape(w_2))

########################################
# Functions
########################################

def bias_expand(vector, dim):
    size = np.shape(vector)
    if dim == 0:
        if len(size) > 1:
            new_vector = np.ones((size[0] + 1, size[1]))
            new_vector[0:size[0], :] = vector
            return new_vector
        else:
            new_vector = np.ones((size[0] + 1,))
            new_vector[0:size[0]] = vector
            return new_vector
    else:
        new_vector = np.ones((size[0], size[1] + 1))
        new_vector[:, 0:size[1]] = vector
        return new_vector

def softmax(vector):
    probs = vector - max(vector)
    probs = np.exp(probs)
    probs = probs / np.sum(probs)
    return probs
    
def loss(probs, label):
    class_label = np.argmax(label)
    return -np.log(probs[class_label])

def update_learning_rate(baseline, T, ep):
    learn_rate = baseline / (1 + ep / T)
    return learn_rate

def sigmoid(vector):
    vec = np.exp(np.multiply(np.ones(np.shape(vector)), vector))
    vec = np.divide(np.ones(np.shape(vec)), np.ones(np.shape(vec)) + vec)
    return vec

def delta(output, label, sigs, weight2):
    delta_2 = label - output
    sig_grad = np.multiply(sigs, np.ones(np.shape(sigs)) - sigs)
    delta_1 = np.multiply(np.matmul(weight2[:,0:-1].T, delta_2), sig_grad[0:-1])
    return (delta_1, delta_2)

def gradient(z_input, delta):
    grad = np.zeros((len(delta), len(z_input)))
    for i in range(len(delta)):
        grad[i, :] = np.multiply(delta[i], z_input) 
    return grad

def forward(data, weight1, weight2):
    data = bias_expand(data, 0)
    hidden_in = np.matmul(weight1, data)
    hidden_out = bias_expand(sigmoid(hidden_in), 0)
    soft_in = np.matmul(weight2, hidden_out)
    soft_out = softmax(soft_in)
    return soft_out, hidden_out, data

def update_weights(gradient, weight1, weight2, dim=None):
    if dim == 0:
        weight1 = weight1 - learn_rate * gradient
        return weight1
    elif dim == 1:
        weight2 = weight2 + learn_rate * gradient
        return weight2

def parameters(data, labels, accuracy=True, losses=False, risk=False):
    data_size = np.shape(data)
    accuracies = np.zeros(10)
    data_loss = -1
    data_risk = -1
    for i in range(data_size[0]):
        output, _, _ = forward(data[i, :], w_1, w_2)
        if accuracy:
            class_label = np.argmax(output)
            gt = np.argmax(labels[i, :])
            if class_label == gt:
                accuracies[gt] = accuracies[gt] + 1
        if losses or risk:
            data_loss = data_loss + loss(output, labels[i, :])
    acc = np.sum(accuracies)
    acc = (1 / data_size[0]) * acc
    if risk:
        data_risk = (1 / data_size[0]) * data_loss
    return acc, data_loss, data_risk
    
# def normalize(weigths):
#     weight_size = np.shape(weights)
    

########################################
# Numerical Gradient Check
########################################

# Generate Sample
sample_index = np.random.randint(0, train_size[1], 1)
sample = mnist.train.images[sample_index, :]
sample = sample[0, :].T
sample_label = mnist.train.labels[sample_index, :]
sample_label = sample_label[0, :].T

# Real Gradient
output, hidden_out, new_sample = forward(sample, w_1, w_2)
deltas = delta(output, sample_label, hidden_out, w_2)
gradient_2 = gradient(hidden_out, deltas[1])
gradient_1 = gradient(new_sample, deltas[0])

# Numerical Gradient 1st Layer

w_1_add = np.array(w_1)
w_1_sub = np.array(w_1)
cnt = 0
for i in range(len(n)):
    for j in range(ep_shape[0]):
        w_1_add[n[i], n[i]] = w_1[n[i], n[i]] + epsilon[j]
        w_1_sub[n[i], n[i]] = w_1[n[i], n[i]] - epsilon[j]
        output_add, _, _ = forward(sample, w_1_add, w_2)
        output_sub, _, _ = forward(sample, w_1_sub, w_2)
        loss_add = loss(output_add, sample_label)
        loss_sub = loss(output_sub, sample_label)
        grad_n = (loss_add - loss_sub) / (2 * epsilon[j])
        errors[j, i + cnt] = np.absolute(gradient_1[n[i], n[i]] - grad_n)
#         print("Real Gradient: {0}".format(str(gradient_1[n[i], n[i]])))
#         print("Numerical Gradient: {0}".format(str(grad_n)))
#         print("Difference: {0}".format(str(errors[j, i + cnt])))
    w_1_add = np.array(w_1)
    w_1_sub = np.array(w_1)
    cnt = cnt + 1
    for j in range(ep_shape[0]):
        w_1_add[n[i], -1] = w_1[n[i], -1] + epsilon[j]
        w_1_sub[n[i], -1] = w_1[n[i], -1] - epsilon[j]
        output_add, _, _ = forward(sample, w_1_add, w_2)
        output_sub, _, _ = forward(sample, w_1_sub, w_2)
        loss_add = loss(output_add, sample_label)
        loss_sub = loss(output_sub, sample_label)
        grad_n = (loss_add - loss_sub) / (2 * epsilon[j])
        errors[j, i + cnt] = np.absolute(gradient_1[n[i], -1] - grad_n)
#         print("Real Gradient: {0}".format(str(gradient_1[n[i], -1])))
#         print("Numerical Gradient: {0}".format(str(grad_n)))
#         print("Difference: {0}".format(str(errors[j, i + cnt])))
    w_1_add = np.array(w_1)
    w_1_sub = np.array(w_1)
    
# Numerical Gradient 2nd Layer

w_2_add = np.array(w_2)
w_2_sub = np.array(w_2)
cnt = 4
for i in range(len(n)):
    for j in range(ep_shape[0]):
        w_2_add[n[i], n[i]] = w_2[n[i], n[i]] + epsilon[j]
        w_2_sub[n[i], n[i]] = w_2[n[i], n[i]] - epsilon[j]
        output_add, _, _ = forward(sample, w_1, w_2_add)
        output_sub, _, _ = forward(sample, w_1, w_2_sub)
        loss_add = loss(output_add, sample_label)
        loss_sub = loss(output_sub, sample_label)
        grad_n = (loss_add - loss_sub) / (2 * epsilon[j])
        errors[j, i + cnt] = np.absolute(gradient_2[n[i], n[i]] + grad_n)
#         print("Real Gradient: {0}".format(str(gradient_2[n[i], n[i]])))
#         print("Numerical Gradient: {0}".format(str(grad_n)))
#         print("Difference: {0}".format(str(errors[j, i + cnt])))
    w_2_add = np.array(w_2)
    w_2_sub = np.array(w_2)
    cnt = cnt + 1
    for j in range(ep_shape[0]):
        w_2_add[n[i], -1] = w_2[n[i], -1] + epsilon[j]
        w_2_sub[n[i], -1] = w_2[n[i], -1] - epsilon[j]
        output_add, _, _ = forward(sample, w_1, w_2_add)
        output_sub, _, _ = forward(sample, w_1, w_2_sub)
        loss_add = loss(output_add, sample_label)
        loss_sub = loss(output_sub, sample_label)
        grad_n = (loss_add - loss_sub) / (2 * epsilon[j])
        errors[j, i + cnt] = np.absolute(gradient_2[n[i], -1] + grad_n)
#         print("Real Gradient: {0}".format(str(gradient_2[n[i], -1])))
#         print("Numerical Gradient: {0}".format(str(grad_n)))
#         print("Difference: {0}".format(str(errors[j, i + cnt])))
    w_2_add = np.array(w_2)
    w_2_sub = np.array(w_2)

#########################################
# Log Log Plots of Error
#########################################

plt.loglog(epsilon, errors[:, 0], epsilon, errors[:, 2])
plt.ylabel("Error")
plt.xlabel("Epsilon")
plt.title("Numerical Errors Weights of Layer 1")
plt.legend(('$w_{0,0}$', '$w_{2,2}$'))
plt.show()

plt.loglog(epsilon, errors[:, 1], epsilon, errors[:, 3])
plt.ylabel("Error")
plt.xlabel("Epsilon")
plt.title("Numerical Errors Biasess of Layer 1")
plt.legend(('$b_{0,0}$', '$b_{2,2}$'))
plt.show()

plt.loglog(epsilon, errors[:, 4], epsilon, errors[:, 6])
plt.ylabel("Error")
plt.xlabel("Epsilon")
plt.title("Numerical Errors Weights of Layer 2")
plt.legend(('$w_{0,0}$', '$w_{2,2}$'))
plt.show()

plt.loglog(epsilon, errors[:, 5], epsilon, errors[:, 7])
plt.ylabel("Error")
plt.xlabel("Epsilon")
plt.title("Numerical Errors Biasess of Layer 2")
plt.legend(('$b_{0,0}$', '$b_{2,2}$'))
plt.show()


#########################################
# Network Training
#########################################

gradient_2 = np.zeros(np.shape(w_2))
gradient_1 = np.zeros(np.shape(w_1))

for i in range(epochs):
    print("Epoch: {0}".format(str(i)))
    for j in range(train_size[0]):
#         print("J: {0}".format(str(j)))   
        output, hidden_out, data = forward(train_images[j, :], w_1, w_2)
        deltas = delta(output, train_labels[j, :], hidden_out, w_2)
        gradient_2 = gradient_2 + gradient(hidden_out, deltas[1])
        gradient_1 = gradient_1 + gradient(data, deltas[0])
        if np.remainder(j, batch) == 0 and j != 0:
            w_2 = update_weights(gradient_2, w_1, w_2, dim=1)
            w_1 = update_weights(gradient_1, w_1, w_2, dim=0)
            gradient_2 = np.zeros(np.shape(w_2))
            gradient_1 = np.zeros(np.shape(w_1))
            accuracies, set_loss, _ = parameters(train_images, train_labels, losses=True)
            print("Loss: {0}".format(str(set_loss)))
            print("Accuracy: {0}".format(str(accuracies)))
            train_loss.append(set_loss)
            train_accuracy.append(accuracies)
    accuracy_epoch[i, 0], _, risk_epoch[i, 0] = parameters(train_images, train_labels, risk=True)
    accuracy_epoch[i, 1], _, risk_epoch[i, 1] = parameters(val_images, val_labels, risk=True)
    if i > 0:
        if accuracy_epoch[i, 1] > accuracy_epoch[i - 1, 1]:
            best_w_1 = w_1
            best_w_2 = w_2
    learn_rate = update_learning_rate(base_rate, T, i)
            


