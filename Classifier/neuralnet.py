import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension
        in_size -> h -> out_size , where  1 <= h <= 256
    

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        #Code goes here
        #USE RELU NOT TANH

        # self.bruh = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1), nn.tanH(), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2), nn.tanH(), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1), nn.tanH(), nn.MaxPool2d(kernel_size=2, stride=2),
        # )
        # self.bruhTheSecond = nn.Sequential(
        #     nn.Linear((128*3*3), 256), nn.tanH(), nn.Dropout(0.5), nn.Linear(256,
        #                                                                      128), nn.tanH(), nn.Dropout(0.5), nn.Linear(128, out_size)
        # )

        #2 Conv layers, 
        #2 Linear Layers
        # 2 dropouts
        transitionChannels = 32
        # dropout = .25
        
        self.cnnFirstPart = nn.Sequential(
            #First conv layer
            nn.Conv2d(in_channels=3, out_channels=transitionChannels, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=transitionChannels, out_channels=64,kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),
        )
        #add one more linear layer if doesn't work
        self.cnnSecondPart = nn.Sequential(
            # Adjust the number of units in the first linear layer
            # nn.Linear(576, 512),nn.ReLU(),
            # nn.Dropout(0.5),  # Add dropout after the first linear layer
            # nn.Linear(512, 256),  # Add a second linear layer
            # nn.ReLU(),
            # nn.Linear(256, out_size)
            nn.Linear(576, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, out_size)
        )

        
        self.optimizer = optim.SGD(self.parameters(), lr=lrate)

    
    def forward(self, x):
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """

        #Why not working
        #change view
        # x = x.view((-1*1), (1*3), 32, 32)??????
        x = x.view((-1*1), (1*3), 31, 31)
        
        x = self.cnnFirstPart(x)
        
        # If you're using a convolutional net, you need to reshape your data in the forward() method, and not the fit() method
        #x = x.view(x.size(0), (-1*1))?
        x = x.view(x.size(0),(-1*1))
        return self.cnnSecondPart(x)
        # raise NotImplementedError("You need to write this part!")
        # return torch.ones(x.shape[0], 1)

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        
        # self.train()
        yhat = self.forward(x)
        loss = self.loss_fn(yhat, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
        # raise NotImplementedError("You need to write this part!")
        # return 0.0


#DO NOT CHANGE fit function
def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """ 
    Make NeuralNet object 'net'. Use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method *must* work for arbitrary M and N.

    @return losses: list of floats containing the total loss at the beginning and after each epoch.
        Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    losses, yhats = [], []
    trainDataSet = get_dataset_from_arrays(train_set, train_labels)
    trainLoad = DataLoader(trainDataSet, batch_size=batch_size, shuffle=False)
    neuralNet = NeuralNet(0.01, nn.CrossEntropyLoss(),in_size=2883, out_size=4)

    for eachEpoch in range(epochs):
        epochLoss = 0.0
        for eachBach in trainLoad:
            inputs, labels = eachBach['features'], eachBach['labels']

            batchLoss = neuralNet.step(inputs, labels)
            epochLoss += batchLoss

        losses.append(epochLoss)


    for eachDev in dev_set:
        yhats.append(torch.argmax(neuralNet.forward(eachDev)))
    
    yhats = np.array(yhats)

    return losses, yhats, neuralNet

    # raise NotImplementedError("You need to write this part!")
    # return [],[],None
