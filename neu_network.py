# Creator LE
# Time 2021/8/29 12:48
# coding=UTF-8

import matplotlib.pyplot
import numpy
import scipy.special
import time

class NeuralNetwork:  # neural network with 3 layers
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # they're all numbers
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes
        self.l_rate = learning_rate

        # weight from input to hidden / from hidden to output
        self.w_ih = numpy.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.w_ho = numpy.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))

        # activation
        self.activation_func = lambda x: scipy.special.expit(x)

    def train_1(self, input_list, target_list):  # 均为横着的list
        target = numpy.array(target_list, ndmin=2).T
        input_ = numpy.array(input_list, ndmin=2).T
        hidden_input = numpy.dot(self.w_ih, input_)
        hidden_output = self.activation_func(hidden_input)
        final_input = numpy.dot(self.w_ho, hidden_output)
        final_output = self.activation_func(final_input)

        loss = target - final_output

        hidden_loss = numpy.dot(self.w_ho.T, loss)

        self.w_ho += self.l_rate * numpy.dot((loss * final_output * (1.0 - final_output)), hidden_output.T)
        self.w_ih += self.l_rate * numpy.dot((hidden_loss * hidden_output * (1.0 - hidden_output)), input_.T)

    def train_2(self, input_list, target_list):  # a more accurate gradient descent one
        target = numpy.array(target_list, ndmin=2).T
        input_ = numpy.array(input_list, ndmin=2).T
        hidden_input = numpy.dot(self.w_ih, input_)
        hidden_output = self.activation_func(hidden_input)
        final_input = numpy.dot(self.w_ho, hidden_output)
        final_output = self.activation_func(final_input)

        loss = target - final_output

        self.w_ih += self.l_rate * numpy.dot(
            numpy.dot(self.w_ho.T, loss * final_output * (1.0 - final_output)) * hidden_output * (1.0 - hidden_output),
            input_.T)
        self.w_ho += self.l_rate * numpy.dot((loss * final_output * (1.0 - final_output)), hidden_output.T)  #  # a

    def query_array(self, input_list):  # 横着的list
        input_ = numpy.array(input_list, ndmin=2).T
        hidden_input = numpy.dot(self.w_ih, input_)
        hidden_output = self.activation_func(hidden_input)
        final_input = numpy.dot(self.w_ho, hidden_output)
        final_output = self.activation_func(final_input)
        return final_output

    def query_int(self, input_list):
        arr = self.query_array(input_list)
        return numpy.argmax(arr)


def display(index):
    test_file = open('mnist_test.csv', "r")
    data_list = test_file.readlines()
    test_file.close()
    all_values = data_list[index].split(',')
    tag = int(all_values[0])
    image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
    matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
    matplotlib.pyplot.show()


def pre_treat(array):
    array = array / 255.0 * 0.99 + 0.01
    return array


def main_train(net, train_size, train_type):
    if train_size > 60000:
        print('Out of range')
        return 1
    train_file = open('mnist_train.csv', 'r')
    data_list = train_file.readlines()
    train_file.close()
    all_value = []
    for i in range(train_size):
        all_value.append((data_list[i].strip('\n')).split(','))
        tag = int(all_value[i][0])
        target = numpy.zeros([1, 10])
        target += 0.01
        target[0][tag] += 0.98
        input_list = numpy.asfarray(all_value[i][1:]).reshape((1, 784))
        if train_type == 1:
            net.train_1(pre_treat(input_list), target)
        elif train_type == 2:
            net.train_2(pre_treat(input_list), target)
    return 0


def test(net, test_size):
    if test_size > 10000:
        print('Out of range')
        return 1
    test_file = open('mnist_test.csv', 'r')
    data_list = test_file.readlines()
    test_file.close()
    all_value = []
    score = 0
    for i in range(test_size):
        all_value.append((data_list[i].strip('\n')).split(','))
        tag = int(all_value[i][0])
        input_list = numpy.asfarray(all_value[i][1:]).reshape((1, 784))
        ans = net.query_int(pre_treat(input_list))
        if tag == ans:
            score += 1
        # else:
        #     matplotlib.pyplot.imshow(input_list.reshape(28,28), cmap='Greys', interpolation='None')
        #     matplotlib.pyplot.show()
        #     print(f"tag:{tag}", f"ans:{ans}")
        #     print('---')
        #     time.sleep(3)

    print('-----------------------')
    print(f'test_size = {test_size}')
    print(f'score     = {score}')
    print(f'accuracy = {float(score) / test_size}')
    return 0


if __name__ == '__main__':
    main_net = NeuralNetwork(784, 300, 10, 0.2)
    main_train(main_net, train_size=30000, train_type=1)
    test(main_net, 1000)
