"""
author: wenzheng dong
----------------------
This module implements a torch-based D-NN to learn the ctrl_env convolution 

"""

# Preamble
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import zipfile    
import os
import pickle

from torch.utils.data import TensorDataset, DataLoader


#pauli_operators = [np.array([[1,0],[0,1]]), np.array([[0,1],[1,0]]), np.array([[0,-1j],[1j,0]]), np.array([[1,0],[0,-1]]) ]
#inital_states = 1/2*np.array([ (pauli_operators[0]+pauli_operators[1]), (pauli_operators[0]-pauli_operators[1]), \
#                               (pauli_operators[0]+pauli_operators[2]), (pauli_operators[0]-pauli_operators[2]),\
#                               (pauli_operators[0]+pauli_operators[3]), (pauli_operators[0]-pauli_operators[3]) ])

#pauli_operators = torch.tensor(pauli_operators)
#inital_states   = torch.tensor(inital_states  )   


############################################################################################################
#########        load data 
############################################################################################################

#f = open(os.path.dirname(__file__)+"/simu_data_Gaussian_L=4.ds", 'rb')
#data = pickle.load(f)
#f.close()

#inputs  = np.array(data["training_inputs"] ,dtype='float32')
#targets = np.array(data["training_targets"],dtype='float32') 


#nInput  = inputs.shape[1]* inputs.shape[2]                      # = L*3
#nOutput = targets.shape[1]* targets.shape[2]                    # =3*6= 18

#print(nInput, nOutput)

############################################################################################################
#########        build NN model 
############################################################################################################


class MyNN(nn.Module):
    """
    define NN model: there are '4' hidden layers
    """
    def __init__(self, nInput=12, nHidden=3, nOutput=18) -> None:
        super().__init__()
        self.layers   = nn.Sequential(
            nn.Linear(nInput, nHidden),         # 1st hidden layer
            nn.Sigmoid(),
            nn.Linear(nHidden,nHidden),         # 2nd hidden layer
            nn.Sigmoid(), 
            nn.Linear(nHidden,nHidden),         # 3rd hidden layer    
            nn.Sigmoid(),
            nn.Linear(nHidden, nHidden),        # 4th hidden layer    
            nn.Sigmoid(),
            nn.Linear(nHidden, nOutput),        # Output layer    
        )
    def forward(self,x):
        return (self.layers(x))

class Training(MyNN):
    """
    A class that train the NN model:
    Input: the 'inputs' & targets (it will be 'tensor'-preprocessed)
    Output:
    """
    def __init__(self, inputs, targets) -> None:
        super().__init__()
        self.inputs     = inputs                                                                  # Input data
        self.targets    = targets                                                                 # Output data
        self.nInputs    = inputs.shape[1] * inputs.shape[2]                                       # nInput based on input data
        self.nOutputs   = targets.shape[1]* targets.shape[2]                                      # nOutput based on outout data
        self.model      = MyNN(nInput = self.nInputs, nHidden=3, nOutput= self.nOutputs)          # instantiate a NN model
        self.opt        = torch.optim.SGD(self.model.parameters(), lr=1e-4)                       # set optimizer of the NN model
        self.criterion  = nn.MSELoss()                                                            # set loss_function of the NN model
        self.losses_training   = []    
        self.losses_testing    = []                                                                      
        self.Epochs     = 300
        self.batch_size = 10 
        self.train_ds   = TensorDataset(torch.reshape(torch.from_numpy(inputs ), (-1,self.nInputs)), \
                                      torch.reshape(torch.from_numpy(targets), (-1,self.nOutputs)))              # Define dataset
        self.train_dl   = DataLoader(self.train_ds, self.batch_size, shuffle=False)                                # Define data loader
                
    def fit(self):
        """
        training the neural network
        """
        for epoch in range(self.Epochs):
            for x, y in self.train_dl:
                #print(x.size, y.size)
                #print(x,y)
                pred = self.model(x)
                loss = self.criterion(pred, y)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            if (epoch + 1) % 100 == 0:
                print('Loss after epoch %5d: %.3f' %(epoch + 1, loss))    
            #np.append(self.losses_training, loss)                                               # log the loss on training set
    
    def test(self, load_opt_name):
        """
        load the saved optimizer from training for testing
        """
        self.opt.load_state_dict(torch.load(load_opt_name))                                     # load the optimizer pt file
        for x,y in self.train_dl:
            pred = self.model(x)
            loss = self.criterion(pred, y)
            #np.append(self.losses_testing, loss)                                                # log the loss on testing set








############################################################################################################
#########        training the NN
############################################################################################################

if __name__ == '__main__':
    
    ################ Gaussian control ####################
    print('start training Gaussian ctrl:')
    f = open(os.path.dirname(__file__)+"/simu_data_Gaussian_L=4.ds", 'rb')
    data = pickle.load(f)
    f.close()

    inputs  = np.array(data["training_inputs"] ,dtype='float32')
    targets = np.array(data["training_targets"],dtype='float32') 

    my_training_G = Training(inputs, targets)                           # Instantiate the class 
    my_training_G.fit()                                                 # training
    torch.save(my_training_G.opt.state_dict(), os.path.dirname(__file__)+"/"+"optimizer_G.pt")        # Save optimizer 
    #//// https://stackoverflow.com/questions/74626924/state-dict-error-for-testing-a-model-that-is-trained-by-loading-checkpoints
    my_training_G.test(os.path.dirname(__file__)+"/optimizer_G.pt")      # test

    #for param_tensor in my_training_G.state_dict():
    #    print(param_tensor, "\t", my_training_G.state_dict()[param_tensor].size())
    #for var_name in my_training_G.opt.state_dict():
    #    print(var_name, "\t", my_training_G.opt.state_dict()[var_name])

    ################ Trigangle control ####################
    print('start training Triangle ctrl:')
    f = open(os.path.dirname(__file__)+"/simu_data_Triangle_L=4.ds", 'rb')
    data = pickle.load(f)
    f.close()

    inputs  = np.array(data["training_inputs"] ,dtype='float32')
    targets = np.array(data["training_targets"],dtype='float32') 

    my_training_T = Training(inputs, targets)                           # Instantiate the class 
    my_training_T.fit()                                                 # training
    torch.save(my_training_T.opt.state_dict(), os.path.dirname(__file__)+"/"+"optimizer_T.pt")        # Save optimizer
    my_training_G.test(os.path.dirname(__file__)+"/optimizer_T.pt")      # test


