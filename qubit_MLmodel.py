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

pauli_operators = [np.array([[1,0],[0,1]]), np.array([[0,1],[1,0]]), np.array([[0,-1j],[1j,0]]), np.array([[1,0],[0,-1]]) ]
inital_states = 1/2*np.array([ (pauli_operators[0]+pauli_operators[1]), (pauli_operators[0]-pauli_operators[1]), \
                               (pauli_operators[0]+pauli_operators[2]), (pauli_operators[0]-pauli_operators[2]),\
                               (pauli_operators[0]+pauli_operators[3]), (pauli_operators[0]-pauli_operators[3]) ])

pauli_operators = torch.Tensor(pauli_operators)
inital_states   = torch.Tensor(inital_states)     

nInput  = 12 # L=4, 3*4=12
nOutput = 18 # 3*6 =18

def U_ctrl_T():
    pass
def ctrl_params_dynamical_programm():
    pass
class QubitControl():
    """
    Process the pulse parameters 
    """
    def __init__(self) -> None:
        super().__init__()
        pass
    def ctrl_params_dynamical_programm():
        """
        use DP to calcualte the accumulated ctrl_unitary
        """
        pass
    def U_ctrl_T():
        """
        """
        pass

class QubitNNmodel(nn.Module):
    """
    the NN-based model to train the qubit
    """
    def __init__(self, inputs, targets, nHidden=3, nOutput=18) -> None:
        """
        inputs  : list of ctrl_params, size = L*3
        targets : list of toggling-frame msmt_results, size=18
        """
        super().__init__()
        self.inputs  = inputs       # x in ML = ctrl_params ;
        self.targets = targets      # y in ML = msmt_results; 
        nInput  = 12           # len(x) in ML      
        self.batch_size = 10 
        self.train_ds = TensorDataset(inputs, targets)                           # Define dataset
        self.train_dl = DataLoader(self.train_ds, self.batch_size, shuffle=True) # Define data loader
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
        )                                                                        # Make the structured-NN       
        self.model    = self.layers(self.inputs)                                 # Define the model
        self.opt      = torch.optim.Adam(self.model.parameters(), lr=1e-3)       # Define optimizer
        self.loss_fn  = nn.MSELoss()                                             # Define loss function
        self.losses   = []                                                       # List of losses in training

    def forward(self,x):
        """
        Convert the 
        """ 
        #x= self.input_regularize(x)
        return (self.layers(x))
    
    def fit(self, ePochs):
        """
        define a utility function to train the model
        """
        for epoch in ePochs:
            for x, y in self.train_dl:
                y_Pred = self.model(x)                # Generate predictions
                loss = self.loss_fn(y_Pred,y)         # measure the loss
                #Perform gradient descent:
                self.loss_fn.backward()
                self.opt.step()
                self.opt.zero_grad()
            #print('Training loss: ', self.loss_fn(self.model(self.x), self.y))
            if (epoch + 1) % 1000 == 0:
                print('Loss after epoch %5d: %.3f' %(epoch + 1, loss))
            # keep track of losses
            self.losses.append(loss.detach().numpy)            
    
    def save_model(self,filename):
        torch.save(self.model.state_dict(), "./")

    def load_model(self,filename):
        
        self.model = torch.load(torch.load("./"))


########################################################################################################
########################################################################################################        
########################################################################################################


    def input_regularize(self,pulse_params):
        """
        regularize the input data using 
        1) ctrl_params_dynamical_programm
        2) normalization & torch_tensors
        """
        pulse_params = ctrl_params_dynamical_programm()
        return torch.tensor(pulse_params, dtype=torch.float32)

    def VoConstructor(self, idx_O):
        """
        idx_O = 0,1,2 = Ox, Oy, Oz
        """
        pass

    def msmt_readout(self,pulse_params,idx_O, idx_S):
        """
        gives the E[O]_S = Tr[Tr[VO(T)ρ_s tilde{O}]]
        """
        # the actual tilde{O}  = U0^+ @ O @ U0 
        weight_R_to_T = [np.trace[(U_ctrl_T() @ pauli_operators[idx_O+1] @ U_ctrl_T().getH()) @ pauli] \
                         for pauli in pauli_operators[1:]]
        Vo = np.sum( [ weight_R_to_T[i] * self.VoConstructor(i)  for i in range(3) ] )  
        O_t = np.sum([weight_R_to_T[i]*pauli_operators[i+1] for i in range(3)])
        return np.trace(Vo @ inital_states[idx_S] @  O_t) 
    
    def criterion(self):
        self.criterion = nn.MSELoss()
    
    def optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.01)
    
    def training_data(self, epochs =100):
        """
        training 
        """
        losses = []
        for i in range(epochs):
            # go forward and get a prediction
            y_pred =self.forward(pulse_params)

            # measure the loss
            loss = self.criterion(y_pred, y_train)

            # keep track of losses
            losses.append(loss.detach().numpy)

            # backprop: take the error rate of forward prop and feed it back thourgh NN to fine tune the weights
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

    def testing_data(self):
        pass
    def save_model(self):
        pass
    def load_model(self):
        pass


class NN_Model(nn.Module):
    """
    a multilayer NN-based model:
    ================================================================================================================
    ||  INPUT_LAYER --> HIDDEN_LAYER_1 --> HIDDEN_LAYER_2 --> HIDDEN_LAYER_3 --> HIDDEN_LAYER_4 --> OUTPUT_LAYER  ||
    ================================================================================================================
    """
    def __init__(self, nInput, nHidden=3, nOutput=18) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(nInput,  nHidden),        # 1st hidden layer
            nn.Sigmoid(),
            nn.Linear(nHidden, nHidden),        # 2nd hidden layer
            nn.Sigmoid(), 
            nn.Linear(nHidden, nHidden),        # 3rd hidden layer    
            nn.Sigmoid(),
            nn.Linear(nHidden, nHidden),        # 4th hidden layer    
            nn.Sigmoid(),
            nn.Linear(nHidden, nOutput),        # Output layer    
        )

    def forward(self,x):
        return (self.layers(x))  

class qubit_ML_model(NN_Model):
    """
    the NN-based model to train the qubit
    """
    def __init__(self, nInput, nHidden=3, nOutput=18) -> None:
        super().__init__(nInput, nHidden, nOutput)
