import os
import matplotlib.pyplot as plt
# %matplotlib inline
plt.style.use('ggplot')

def train_model_and_plot_stats(
        model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True):
    
    # As well as monitoring the error over training also monitor classification
    # accuracy i.e. proportion of most-probable predicted classes being equal to targets
    data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

    # Use the created objects to initialise a new Optimiser instance.
    optimiser = Optimiser(
        model, error, learning_rule, train_data, valid_data, data_monitors, notebook=notebook)

    # Run the optimiser for num_epochs epochs (full passes through the training set)
    # printing statistics every epoch.
    stats, keys, run_time = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)

    return stats, keys, run_time

# The below code will set up the data providers, random number
# generator and logger objects needed for training runs. As
# loading the data from file take a little while you generally
# will probably not want to reload the data providers on
# every training run. If you wish to reset their state you
# should instead use the .reset() method of the data providers.
import numpy as np
import logging
import sys
# sys.path.append('/path/to/mlpractical')
from mlp.data_providers import MNISTDataProvider, EMNISTDataProvider

# Seed a random number generator
seed = 11102019 
rng = np.random.RandomState(seed)
batch_size = 100
# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# The model set up code below is provided as a starting point.
# You will probably want to add further code cells for the
# different experiments you run.

# %pip install tqdm

from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer, ReluLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import AdamLearningRule
from mlp.optimisers import Optimiser

num_epochs = 100

def Experiment_Width(widths=[32, 64, 128]):

    stats_list = []

    # Setup hyperparameters
    learning_rate = 9e-4
    stats_interval = 1

    for width in widths:

        input_dim, output_dim, hidden_dim = 784, 47, width

        # Create data provider objects for the MNIST data set
        train_data = EMNISTDataProvider('train', batch_size=batch_size, rng=rng)
        valid_data = EMNISTDataProvider('valid', batch_size=batch_size, rng=rng)

        # re-init params
        weights_init = GlorotUniformInit(rng=rng)
        biases_init = ConstantInit(0.)

        # Create model with ONE hidden layer
        model = MultipleLayerModel([
            AffineLayer(input_dim, hidden_dim, weights_init, biases_init), # hidden layer
            ReluLayer(),
            AffineLayer(hidden_dim, output_dim, weights_init, biases_init) # output layer
        ])

        error = CrossEntropySoftmaxError()
        # Use a Adam learning rule
        learning_rule = AdamLearningRule(learning_rate=learning_rate)

        print(f"---------------------------------------------------------------------------------------------")

        print(f"------------------------- Total Epoch: {num_epochs}, Width: {width} -------------------------")

        print(f"---------------------------------------------------------------------------------------------")

        # Remember to use notebook=False when you write a script to be run in a terminal
        stats, keys, run_time = train_model_and_plot_stats(
            model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=False)
        
        stats_list.append(stats)

    # Plot the change in the validation and training set error over training.
    fig_1 = plt.figure(figsize=(8, 4))
    ax_1 = fig_1.add_subplot(111)
    
    for k in ['error(train)', 'error(valid)']:

        if k == 'error(train)':
                linestyle = "-"
                label = "(train)"
        else: 
            linestyle = "--"
            label = "(valid)"

        for idx, stats in enumerate(stats_list):

            print(f"Width: {widths[idx]}, Final error{label}: {stats[-1, keys[k]]}")

            ax_1.plot(np.arange(1, stats.shape[0]) * stats_interval, 
                    stats[1:, keys[k]], label=f"width {widths[idx]}{label}", linestyle=linestyle)
            
            
    ax_1.legend(loc=0)
    ax_1.set_xlabel('Epoch number')
    ax_1.set_ylabel('Error')

    plt.savefig("./report/figures/Task1_error_curve_width.pdf")

    # Plot the change in the validation and training set accuracy over training.
    fig_2 = plt.figure(figsize=(8, 4))
    ax_2 = fig_2.add_subplot(111)

    for k in ['acc(train)', 'acc(valid)']:

        if k == 'acc(train)':
                linestyle = "-"
                label = "(train)"
        else: 
            linestyle = "--"
            label = "(valid)"

        for idx, stats in enumerate(stats_list):

            print(f"Width: {widths[idx]}, Final acc{label}: {stats[-1, keys[k]]}")

            ax_2.plot(np.arange(1, stats.shape[0]) * stats_interval, 
                    stats[1:, keys[k]], label=f"width {widths[idx]}{label}", linestyle=linestyle)
        
    ax_2.legend(loc=0)
    ax_2.set_xlabel('Epoch number')
    ax_2.set_ylabel('Accuracy')

    plt.savefig("./report/figures/Task1_acc_curve_width.pdf")

    # Write result
    result_path = "./cw1/result/Width/"

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    file_path = os.path.join(result_path, "log.txt")

    with open(file_path, "a") as f:

        for idx, stats in enumerate(stats_list):

            error = []
            acc = []

            for k in ['error(train)', 'error(valid)']:

                print(k, stats[-1, keys[k]])    
                error.append(stats[-1, keys[k]])

            for k in ['acc(train)', 'acc(valid)']:
                    
                print(k, stats[-1, keys[k]])
                acc.append(stats[-1, keys[k]])

            f.write(f"Epoch: {num_epochs}, " + "Width, " + f"width = {widths[idx]}\n" + f"Validation Acc: {np.round(100 * acc[1], 1)}, " + f"Train Error: {np.around(error[0], 3)}, " + f"Validation Error: {np.around(error[1], 3)} \n")

Experiment_Width()

def Experiment_Depth(depths=[1, 2, 3]):

    stats_list = []

    # Setup hyperparameters
    learning_rate = 9e-4
    stats_interval = 1
    input_dim, output_dim, hidden_dim = 784, 47, 128

    for depth in depths:

        # Create data provider objects for the MNIST data set
        train_data = EMNISTDataProvider('train', batch_size=batch_size, rng=rng)
        valid_data = EMNISTDataProvider('valid', batch_size=batch_size, rng=rng)

        # re-init params
        weights_init = GlorotUniformInit(rng=rng)
        biases_init = ConstantInit(0.)

        model_list = [AffineLayer(input_dim, hidden_dim, weights_init, biases_init), ReluLayer()] # init
        for _ in range(depth-1):
            model_list.append(AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init))
            model_list.append(ReluLayer())
        model_list.append(AffineLayer(hidden_dim, output_dim, weights_init, biases_init))

        print(model_list)

        # Create model with TWO hidden layers
        model = MultipleLayerModel(model_list)

        error = CrossEntropySoftmaxError()
        # Use a Adam learning rule
        learning_rule = AdamLearningRule(learning_rate=learning_rate)

        print(f"---------------------------------------------------------------------------------------------")

        print(f"------------------------- Total Epoch: {num_epochs}, Didth: {depth} -------------------------")

        print(f"---------------------------------------------------------------------------------------------")

        # Remember to use notebook=False when you write a script to be run in a terminal
        stats, keys, run_time = train_model_and_plot_stats(
            model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=False)
        
        stats_list.append(stats)

    # Plot the change in the validation and training set error over training.
    fig_1 = plt.figure(figsize=(8, 4))
    ax_1 = fig_1.add_subplot(111)
    
    for k in ['error(train)', 'error(valid)']:

        if k == 'error(train)':
                linestyle = "-"
                label = "(train)"
        else: 
            linestyle = "--"
            label = "(valid)"

        for idx, stats in enumerate(stats_list):

            print(f"Depth: {depths[idx]}, Final error{label}: {stats[-1, keys[k]]}")

            ax_1.plot(np.arange(1, stats.shape[0]) * stats_interval, 
                    stats[1:, keys[k]], label=f"depth {depths[idx]}{label}", linestyle=linestyle)
            
            
    ax_1.legend(loc=0)
    ax_1.set_xlabel('Epoch number')
    ax_1.set_ylabel('Error')

    plt.savefig("./report/figures/Task1_error_curve_depth.pdf")

    # Plot the change in the validation and training set accuracy over training.
    fig_2 = plt.figure(figsize=(8, 4))
    ax_2 = fig_2.add_subplot(111)

    for k in ['acc(train)', 'acc(valid)']:

        if k == 'acc(train)':
                linestyle = "-"
                label = "(train)"
        else: 
            linestyle = "--"
            label = "(valid)"

        for idx, stats in enumerate(stats_list):

            print(f"Depth: {depths[idx]}, Final acc{label}: {stats[-1, keys[k]]}")

            ax_2.plot(np.arange(1, stats.shape[0]) * stats_interval, 
                    stats[1:, keys[k]], label=f"depth {depths[idx]}{label}", linestyle=linestyle)
        
    ax_2.legend(loc=0)
    ax_2.set_xlabel('Epoch number')
    ax_2.set_ylabel('Accuracy')

    plt.savefig("./report/figures/Task1_acc_curve_depth.pdf")

    # Write result
    result_path = "./cw1/result/Depth/"

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    file_path = os.path.join(result_path, "log.txt")

    with open(file_path, "a") as f:

        for idx, stats in enumerate(stats_list):

            error = []
            acc = []

            for k in ['error(train)', 'error(valid)']:

                print(k, stats[-1, keys[k]])    
                error.append(stats[-1, keys[k]])

            for k in ['acc(train)', 'acc(valid)']:
                    
                print(k, stats[-1, keys[k]])
                acc.append(stats[-1, keys[k]])

            f.write(f"Epoch: {num_epochs}, " + "Depth, " + f"depth = {idx+1}\n" + f"Validation Acc: {np.round(100 * acc[1], 1)}, " + f"Train Error: {np.around(error[0], 3)}, " + f"Validation Error: {np.around(error[1], 3)} \n")

Experiment_Depth()
