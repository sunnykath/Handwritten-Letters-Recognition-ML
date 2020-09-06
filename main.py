import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
from sklearn import metrics
import string

import matplotlib.pyplot as plt

from models.MLP import MultiP
from models.AlexNet import AlexNet
from models.LeNet import LeNet
from models.LeNet_ReLU import LeNet_ReLU


def main():

    ########## Choose the model ##########

    network = 'MLP'
    # network = 'LeNet'
    # network = 'AlexNet'
    # network = 'LeNet_ReLU'


    ########## Build a Network and define some HyperParameters ##########
    
    # Initialise the hyper parameters
    batch_size = 100
    rate = 0.001
    MLP = False

    if network == 'MLP':
        model = MultiP(28 * 28, 584, 384, 26)
        MLP = True

    elif network == 'LeNet':
        model = LeNet()

    elif network == 'AlexNet':
        model = AlexNet()

    elif network == 'LeNet_ReLU':
        model = LeNet_ReLU()
        # Fine Tune the hyper parameters to optimise the network
        batch_size = 250
        rate = 0.005


    ########## Import & load the training dataset ##########

    # Set transform(s)
    transform = transforms.Compose([transforms.ToTensor()])

    # Load in the dataset
    train_set = torchvision.datasets.EMNIST(
        './dataset'
        , 'letters'
        , train = True
        , transform = transform
        , download = True
    )


    # Splitting the dataset into batches from the training dataset
    train_loader = torch.utils.data.DataLoader(
        train_set
        , batch_size= batch_size
        , shuffle=True
    )


    ########## Import & load the testing dataset ##########

    # Load in the dataset
    test_set = torchvision.datasets.EMNIST(
        './dataset'
        , 'letters'
        , train=False
        , transform=transform
    )

    # Splitting the dataset into batches from the testing dataset
    test_loader = torch.utils.data.DataLoader(
        test_set
        , batch_size=batch_size
        , shuffle=True
    )


    ### Displaying a BATCH from the dataset with it's corresponding labels
    # for batch in train_loader:
    #     images, labels = batch
    #     break
    # grid = torchvision.utils.make_grid(images)
    # grid_matrix = grid.numpy()
    # # display ⬇⬇
    # print(labels)
    # plt.imshow(grid_matrix.T)
    # plt.show()


    ########## Run the network ##########

    epochs = 25    # define the number of epochs

    # Print details for the network
    print('Using: {}  Batch Size: {}  Learning Rate: {}' .format(network, batch_size, rate))

    # Initialise the optimiser
    optimiser = optim.Adam(model.parameters(), lr= rate)

    # Book Keeping - Initialise the array
    training_accuracy = []

    sample_size = len(train_set)    # Store the total sample size in a variable

    print("Total training samples:", sample_size)

    # Store the starting system time of the training for tracking
    start_time = time.time()

    for epoch in range(epochs):  # Training the model for 'epochs' number of times

        accuracy = train(model, train_loader, optimiser, epoch, sample_size, MLP)

        # Book Keeping - Store the accuracy in an array (y axis)
        training_accuracy.append(accuracy)


    # Calculate and store the elapsed time from the start of the training
    elapsed_time = time.time() - start_time

    # Print the time taken to train the neural network, if needed
    print("Time to train: {} \n" .format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

    # Test the model on the test dataset
    print("Total testing samples:", len(test_set))
    test_accuracy, actual_labels, predictions = test(model, test_loader, len(test_set), MLP)

    print(test_accuracy)


    ########## Plot using matplotlib ##########

    # Book Keeping and formatting time for plotting - Time in minutes (x-Axis)
    # Store the time recorded(in seconds, float) in a list (in milliseconds, int, for better accuracy)
    # Define the step size so that the size of the array is the same as the number of epochs
    plot_time = (list(range(0, int(elapsed_time * 1000), int(elapsed_time * 1000 / (epochs - 1)))))
    plot_time = np.asarray(plot_time) / (60 * 1000)  # Convert the list into array & milliseconds into minutes


    # Plot Formatting
    plt.style.use('dark_background')        # Set the style
    plt.rcParams.update({'font.size': 20})  # Set the font size for the labels

    # Define the figure
    fig, time_axis = plt.subplots(1, 1)
    epochs_axis = time_axis.twiny()         # Same y-axis for the epochs axis

    # Title & Labels
    plt.title("Learning Accuracy Curve", pad=20)    # Plot label
    time_axis.set_xlabel('Time(min)')
    time_axis.set_ylabel('Accuracy(%)')
    epochs_axis.set_xlabel('Epochs')

    # Plot data on their respective axes
    time_axis.plot(plot_time, training_accuracy)
    epochs_axis.plot(range(0, epochs), training_accuracy, 'r', linewidth=2, label='Training Accuracy')
    plt.plot(epochs-1, test_accuracy, 'bX', markersize=15, label='Test Accuracy')   # Plot the test accuracy as well

    # Add a caption for more details on the graph
    graph_text = "Network: {} | Learning Rate: {} | Batch Size: {}".format(network, rate, batch_size)
    plt.text(0, accuracy, graph_text,  style='italic', fontsize=15)

    plt.legend(loc="lower right")

    # Set the figure size and save file
    fig.set_size_inches(15, 10)             # Width, Height
    filename = 'Evaluation/' + network + '.png'
    plt.savefig(filename)

    # Also create and save the confusion matrix for evaluation
    save_confusion_matrix(actual_labels, predictions, network)


###########         Define the training function            ###########

def train(model, train_loader, optimiser, epoch, samples, MLP):

    # Initialise variables
    total_loss = 0
    total_correct = 0
    accuracy = 0


    # Record the time at the start of each epoch
    start_epoch_time = time.time()

    # go through every batch in training subset i.e 15000 samples, 100 samples per batch, 150 batches to iterate through
    for batch in train_loader:
        images, labels = batch
        labels = labels - 1         # indexing error

        # Images need to be flattened for Multilayer Perceptron
        if MLP:
            images = images.view(images.shape[0], -1)

        outputs = model(images)  # pass the images to the network

        # combines softmax() and NLLLoss() functions to calculate the diff between..
        # the correct classes and the current "prediction" to calculate the loss
        loss = F.cross_entropy(outputs, labels)

        # Pytorch adds new gradient to existing gradient, this function zero's the old gradients
        optimiser.zero_grad()

        loss.backward()         # back propagation to calculate the gradient
        optimiser.step()        # update weights to have a smaller gradient or step towards local minima

        total_loss += loss.item()
        total_correct += get_num_correct(outputs, labels)
        accuracy = (total_correct / samples) * 100      # In percentage (%)

    # Calculate and store the elapsed time from the start of the epoch training
    elapsed_epoch_time = time.time() - start_epoch_time

    # Print the results
    print("Epoch: {}  Total Correct: {}/{}  Loss: {:.2f}  Accuracy: {:.0f}%  Time: {}". format(epoch + 1,
        total_correct, samples, total_loss, accuracy, time.strftime("%H:%M:%S", time.gmtime(elapsed_epoch_time))))

    # return the accuracy for book keeping
    return accuracy

###########         Define the Testing function            ###########

def test(model, test_loader, samples, MLP):

    # Initialise the variables
    total_loss = 0
    total_correct = 0
    accuracy = 0
    predictions = torch.tensor([])
    actual_labels = torch.tensor([])

    # go through every batch in testing subset i.e 5000 samples, 100 samples per batch, 50 batches to iterate through
    for batch in test_loader:
        images, labels = batch
        labels = labels - 1         # indexing error

        # Images need to be flattened for Multilayer Perceptron
        if MLP:
            images = images.view(images.shape[0], -1)

        outputs = model(images)  # pass images to the network

        # combines softmax() and NLLLoss() functions to calculate the diff between the correct class
        # and the current "prediction" to calculate the loss
        loss = F.cross_entropy(outputs, labels)

        total_loss += loss.item()
        total_correct += get_num_correct(outputs, labels)
        accuracy = (total_correct / samples) * 100

        # Book Keep all the labels and predictions for the confusion matrix
        labels = torch.as_tensor(labels)
        actual_labels = torch.cat((actual_labels.float(), labels.float()), 0)
        predictions = torch.cat((predictions.float(), outputs.argmax(dim=1).float()), 0)

    print("Test: Total Correct: {}/{}  Loss: {:.02f}  Accuracy: {:.0f}%".format(total_correct, samples, total_loss, accuracy))

    # return some variables for book keeping
    return accuracy, actual_labels, predictions


# Define a function to calculate the number of correct predictions from the model
def get_num_correct(preds, labels):
    # search through activations for the highest value to find the predictions
    predictions = torch.argmax(preds, dim=1)

    # Find the 'correct' predictions by checking them against the 'labels'
    correct = torch.eq(predictions, labels)
    correct = correct.sum().item()          # Sum the correct predictions

    return correct


def save_confusion_matrix(actual_labels, predictions, network):

    # Create the confusion matrix using the sklearn library
    cm = metrics.confusion_matrix(actual_labels, predictions)

    # Plot it on a different figure
    fig2 = plt.figure(2)
    cm_plot = fig2.subplots(1, 1)
    cm_plot.matshow(cm, cmap='OrRd')

    # Manually Set the axes
    cm_plot.set_xticks(range(0, 26))
    cm_plot.set_yticks(range(0, 26))
    cm_plot.set_xticklabels(list(string.ascii_uppercase))
    cm_plot.set_yticklabels(list(string.ascii_uppercase))

    # Labels and Title
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Set a size and save the matrix
    fig2.set_size_inches(20, 20)
    plt.savefig("Evaluation/confusionMatrix_" + network + ".png")

    # Define classes and Create the classification report
    classes = list(string.ascii_uppercase)
    report = metrics.classification_report(actual_labels, predictions, digits=3, target_names=classes)

    # Print the confusion matrix and the report onto a text file
    with open('Evaluation/cm_report_' + network + '.txt', 'w') as file:
        print('Network: ' + network + '\n\rConfusion Matrix:\n', cm, '\n\r\nReport:\n', report, file=file)




if __name__ == '__main__':
    main()
