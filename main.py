import numpy as np
# scipy.special for the sigmoid function expit()
import scipy.special
import matplotlib.pyplot as plt
import time

import scipy.misc
#import glob
import imageio

start_time = time.time()


class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        #set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        #link weight matrices, wih (w_input_hidden) and who (w_hidden_output)
        self.wih = np.random.normal(
            0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(
            0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        # learning rate
        self.lr = learningrate
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr * \
            np.dot((output_errors*final_outputs*(1.0 - final_outputs)),
                   np.transpose(hidden_outputs))
        self.wih += self.lr * \
            np.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),
                   np.transpose(inputs))
        pass

    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


input_nodes = 784
hidden_nodes = 500
output_nodes = 10

learning_rate = 0.2

#create instance of neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open('C:/Users/Lenovo/Desktop/mnist_train_100.csv', 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 2
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:])/255.0*0.99)+0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(record[0])] = 0.99
        n.train(inputs, targets)
        pass
if __name__ == '__main__':
    start_time = time.time()
    print('running...')
    image_file_name = (r'C:/Users/Lenovo/Desktop/8.png')
#    img_array = scipy.misc.imread(image_file_name, flatten = True)
    #print('loading ... ',image_file_name)
    img_array = imageio.imread(image_file_name, as_gray=True)
    img_data = 255.0 - img_array.reshape(784)
    img_data = (img_data/255.0*0.99) + 0.01
    plt.imshow(img_data.reshape((28, 28)), cmap='Greys')
    inputs = img_data
    outputs = n.query(inputs)
    print(outputs)
    label = np.argmax(outputs)  # argmax返回最大值的索引值
    print("network's answer ==>> ", label)
    end_time = time.time()
    print('running time', end_time-start_time)

