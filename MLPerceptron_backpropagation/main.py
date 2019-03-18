import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import randrange, random
import math

list_sum_error = list()


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    foldke = 0
    scores_test = list()
    scores_train = list()
    folds = cross_validation_split(dataset, n_folds)
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            row_copy.pop(-1)
            test_set.append(row_copy)
        print(test_set)
        predicted_test, predicted_train = algorithm(train_set, test_set, foldke, *args)
        actual = [row[-1] for row in fold]
        print(actual)
        print(predicted_test)
        accuracy_test = accuracy_metric(actual, predicted_test)
        accuracy_train = accuracy_metric(actual, predicted_train)
        scores_test.append(accuracy_test)
        scores_train.append(accuracy_train)
        foldke += 1
    return scores_test, scores_train

def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def back_propagation(train, test, foldke, l_rate, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, foldke, l_rate, n_epoch, n_outputs)
    predictions_test = list()
    predictions_train = list()
    for row in test:
        prediction = predict(network, row)
        predictions_test.append(prediction)
    for row in train:
        prediction = predict(network, row)
        predictions_train.append(prediction)
    return predictions_test, predictions_train

def train_network(network, train, foldke,l_rate, n_epoch, n_outputs):
    del list_sum_error[:]
    for epoch in range(n_epoch):
        sum_error = 0
        sum_accuracy = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0.0,1.0,2.0]
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        list_sum_error.append(sum_error)

def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']

def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

def predict(network, row):
    outputs = forward_propagate(network, row)
    print(outputs)
    return outputs.index(max(outputs))

def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation

def transfer(activation):
    return 1.0 / (1.0 + math.exp(-activation))

def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

def transfer_derivative(output):
    return output * (1.0 - output)


if __name__ == '__main__':
    dfdataset = pd.read_csv('iris150.csv').values.tolist()
    nfolds = 5; lrate = 0.1; nepoch =250; nhidden = 5
    scores_test, scores_train = evaluate_algorithm(dfdataset,back_propagation, nfolds, lrate, nepoch, nhidden)
    print(scores_test)
    print(scores_train)
    listaccuracy = np.arange(5)
    plt.plot(listaccuracy,scores_train,'r')
    plt.plot(listaccuracy,scores_test,'b')
    plt.xlabel('fold ke')
    plt.ylabel('scores')
    plt.title('grafik accuracy testset terhadap predict')
    plt.show()
    listepoch = np.arange(nepoch)
    plt.plot(listepoch, list_sum_error)
    plt.xlabel('epoch')
    plt.ylabel('error tiap epoch')
    plt.title('grafik error terhadap epoch fold ke-5')
    plt.show()