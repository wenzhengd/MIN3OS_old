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
from os.path import abspath, dirname, join
import pickle
from math import prod
import scipy.optimize as opt
from joblib import Parallel, delayed

from torch.utils.data import TensorDataset, DataLoader

n_cores = 6


#pauli_operators = [np.array([[1,0],[0,1]]), np.array([[0,1],[1,0]]), np.array([[0,-1j],[1j,0]]), np.array([[1,0],[0,-1]]) ]
#inital_states = 1/2*np.array([ (pauli_operators[0]+pauli_operators[1]), (pauli_operators[0]-pauli_operators[1]), \
#                               (pauli_operators[0]+pauli_operators[2]), (pauli_operators[0]-pauli_operators[2]),\
#                               (pauli_operators[0]+pauli_operators[3]), (pauli_operators[0]-pauli_operators[3]) ])

#pauli_operators = torch.tensor(pauli_operators)
#inital_states   = torch.tensor(inital_states  )   


############################################################################################################
#########        Prepare data 
############################################################################################################

class QubitMeasData():
    """
    Custom 'Dataset' object for regression data.
    Must implement these functions: __init__, __len__, and __getitem__.
    """
    def __init__(self, x,y) -> None:
        self.x = torch.reshape(x, (len(x), -1))
        self.y = torch.reshape(y, (len(y), -1))
    def __getitem__(self, idx):
        return(self.x[idx], self.y[idx])
    def __len__(self, idx):
        return( len(self.x) )


############################################################################################################
#########        build NN model 
############################################################################################################


class MyNN(nn.Module):
    """
    define NN model: there are '4' hidden layers
    """
    def __init__(self, nInput=12, nHidden=4, nOutput=18) -> None:
        super().__init__()
        self.layers   = nn.Sequential(
            nn.Linear(nInput, nHidden),         # 1st hidden layer
            nn.ReLU(),
            nn.Linear(nHidden,nHidden),         # 2nd hidden layer
            nn.ReLU(), 
            nn.Linear(nHidden,nHidden),         # 3rd hidden layer    
            nn.ReLU(),
            nn.Linear(nHidden, nHidden),        # 4th hidden layer    
            nn.ReLU(),
            nn.Linear(nHidden, nHidden),        # 5th hidden layer    
            nn.ReLU(),
            nn.Linear(nHidden, nOutput),        # Output layer
            #nn.Tanh(),                          # Output layer : in range (-1, 1)  
        )
    def my_trig_layer(self,x):
        """
        apply the trignometric function to the input ctrl_params
        """
        return torch.cat((x, torch.sin(x), torch.cos(x), torch.sin(2*x), torch.cos(2*x)), dim=1 )    
    def forward(self,x):
        return (self.layers(self.my_trig_layer(x)))
    
class MyTrigLayer(nn.Module):
    """
    apply the trignometric function to the input ctrl_params
    """    
    def forward(self, x):
        return None

class MyGRU(nn.Module):
    """
    define GRU model: 
    """
    def __init__(self, nInput=3, nHidden=4, nLayer=2, nOutput=18):
        super(MyGRU, self).__init__()
        self.hidden_size = nHidden
        self.num_layers = nLayer
        
        # Define the GRU layer
        self.gru = nn.GRU(nInput, nHidden, nLayer, batch_first=True)
        
        # Define the output layer
        self.fc = nn.Linear(nHidden, nOutput)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate through the GRU layer
        out, _ = self.gru(x, h0)
        
        # Take the output from the last time step
        out = self.fc(out[:, -1, :])
        return out    

class QubitNeuralNetwork(MyNN):
    """
    A class that train the NN model:
    Input: the 'inputs' & targets (it will be 'tensor'-preprocessed)
    Output:
    """
    def __init__(self, inputs, targets) -> None:
        super().__init__()
        self.inputs     = inputs                                                                  # Input data
        self.targets    = targets                                                                 # Output data
        self.num_sample = inputs.shape[0]                                                         # numer of data
        self.nInputs    = inputs.shape[1] * inputs.shape[2]                                       # nInput based on input data (L*3)
        self.nOutputs   = targets.shape[1]* targets.shape[2]                                      # nOutput based on outout data (3*6=12)
        self.model      = MyNN(nInput  = self.nInputs *5, 
                               nHidden =24, 
                               nOutput = self.nOutputs)                                           # Construct a NN model
        self.opt        = torch.optim.SGD(self.model.parameters(), lr=1e-4)                       # set optimizer of the NN model
        self.criterion  = nn.MSELoss()                                                            # set loss_function of the NN model
        self.losses_training   = []    
        self.losses_testing    = []                                                                      
        self.Epochs     = 1000
        self.batch_size = 10 
        self.split      = int(0.75 * self.num_sample)
        # training data:
        self.train_ds   = TensorDataset(torch.reshape(torch.from_numpy(inputs[0:self.split] ), (-1,self.nInputs)), \
                                        torch.reshape(torch.from_numpy(targets[0:self.split]), (-1,self.nOutputs)))              
        self.train_dl   = DataLoader(self.train_ds, self.batch_size, shuffle=False)                    
        # testing data:                                              
        self.testn_ds   = TensorDataset(torch.reshape(torch.from_numpy(inputs[self.split:] ),  (-1,self.nInputs)), \
                                        torch.reshape(torch.from_numpy(targets[self.split:]),  (-1,self.nOutputs)))             
        self.testn_dl   = DataLoader(self.testn_ds, self.batch_size, shuffle=False)                                                                  
    
    def get_data_shape(self):
        """
        get the shape of data in training & testn
        """
        print('nInput, nOutputs -->',self.nInputs, self.nOutputs)
        print("x_train, y_train =", torch.reshape(torch.from_numpy(inputs[0:self.split] ), (-1,self.nInputs)).shape,
                                    torch.reshape(torch.from_numpy(targets[0:self.split]), (-1,self.nOutputs)).shape)    
        print("x_testn, y_testn =", torch.reshape(torch.from_numpy(inputs[self.split:] ), (-1,self.nInputs)).shape,
                                    torch.reshape(torch.from_numpy(targets[self.split:]), (-1,self.nOutputs)).shape)
    def train(self):
        """
        training the neural network
        """
        #print('nInput, nOutputs -->',self.nInputs, self.nOutputs)
        #print('TensorData_train, TensorData_testn -->', self.train_ds.shape,  self.testn_ds.shape)
        for epoch in range(self.Epochs):
            for x, y in self.train_dl:
                #print(x.size, y.size)
                #print(x,y)
                self.opt.zero_grad()
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                self.opt.step()
            if (epoch + 1) % 500 == 0:
                print('Loss after epoch %5d: %.3f' %(epoch + 1, loss))    
            #np.append(self.losses_training, loss)                                               # log the loss on training set
    
    def test(self, load_opt_name):
        """
        load the saved optimizer from training for testing
        """
        self.opt.load_state_dict(torch.load(load_opt_name))                                     # load the optimizer pt file
        for x,y in self.testn_dl:
            pred = self.model(x)
            loss = self.criterion(pred, y)
            self.losses_testing = np.append(self.losses_testing, loss.detach().item())          # log the loss on testing set
        return self.losses_testing 
    
    def optimize_ctrl(self, load_opt_name):
        """
        Fix the NN's weights; optimize the input ctrl_params to 
        obtain the P* that gives best fidelity
        """
        self.opt.load_state_dict(torch.load(load_opt_name))                                     # load the optimizer pt file
        def infidelity(ctl_params):
            """
            ctrl_params are aray with dim=(L,3)
            """
            # sum(Parallel(n_jobs=n_cores, verbose=0)(delayed(devi_sigma_O_T)(o,o,qubit_params,bath_params)  for  o in  _O_ ))
            return self.model(np.array(ctl_params ,dtype='float32').flatten())

        _opt_ = opt.minimize(fun = infidelity,x0= [np.pi, 0,0] * inputs.shape[1] , method ='Nelder-Mead',options={'maxiter': 1000})
        return  _opt_.x


class QubitRecurrentNetwork(MyGRU):
    """
    A class that train the RNN -GRU actually- model:
    Input: the 'inputs' & targets (it will be 'tensor'-preprocessed)
    Output:
    """
    def __init__(self, inputs, targets) -> None:
        super().__init__()
        self.inputs     = inputs                                                                  # Input data
        self.targets    = targets                                                                 # Output data
        self.num_sample = inputs.shape[0]                                                         # numer of data
        self.SequenLen  = inputs.shape[1]                                                         # sequence length = L (=4)
        self.nInputs    = inputs.shape[2]                                                         # nInput based on input data (=3)
        self.nOutputs   = targets.shape[1]*targets.shape[2]*targets.shape[3]                      # nOutput based on outout data (4*3*6=72)
        self.model      = MyGRU(nInput=self.nInputs, 
                                nHidden=10, 
                                nLayer=2, 
                                nOutput= self.nOutputs)                                           # Construct a GRU model
        self.opt        = torch.optim.SGD(self.model.parameters(), lr=1e-4)                       # set optimizer of the NN model
        self.criterion  = nn.MSELoss()                                                            # set loss_function of the NN model
        self.losses_training   = []    
        self.losses_testing    = []                                                                      
        self.Epochs     = 1000
        self.batch_size = 10 
        self.split      = int(0.75 * self.num_sample)
        # training data: 
        self.train_ds   = TensorDataset(torch.from_numpy(inputs[0:self.split]), \
                                        torch.reshape(torch.from_numpy(targets[0:self.split]), (-1,self.nOutputs)))              
        self.train_dl   = DataLoader(self.train_ds, self.batch_size, shuffle=False)                    
        # testing data:                                              
        self.testn_ds   = TensorDataset(torch.from_numpy(inputs[self.split:]), \
                                        torch.reshape(torch.from_numpy(targets[self.split:]),  (-1,self.nOutputs)))             
        self.testn_dl   = DataLoader(self.testn_ds, self.batch_size, shuffle=False)                                                                  
  
    def get_data_shape(self):
        """
        get the shape of data in training & testn
        """
        print('nInput, nOutputs -->',self.nInputs, self.nOutputs)
        print("x_train, y_train =", torch.from_numpy(inputs[0:self.split]).shape,
                                    torch.reshape(torch.from_numpy(targets[0:self.split]), (-1,self.nOutputs)).shape)    
        print("x_testn, y_testn =", torch.from_numpy(inputs[self.split:]).shape,
                                    torch.reshape(torch.from_numpy(targets[self.split:]),  (-1,self.nOutputs)).shape)                
    def train(self):
        """
        training the neural network
        """
        #print('nInput, nOutputs -->',self.nInputs, self.nOutputs)
        #print('TensorData_train, TensorData_testn -->', self.train_ds.shape,  self.testn_ds.shape)
        for epoch in range(self.Epochs):
            for x, y in self.train_dl:
                #print(x.size, y.size)
                #print(x,y)
                self.opt.zero_grad()
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                self.opt.step()
            if (epoch + 1) % 500 == 0:
                print('Loss after epoch %5d: %.3f' %(epoch + 1, loss))    
            #np.append(self.losses_training, loss)                                               # log the loss on training set
    
    def test(self, load_opt_name):
        """
        load the saved optimizer from training for testing
        """
        self.opt.load_state_dict(torch.load(load_opt_name))                                     # load the optimizer pt file
        for x,y in self.testn_dl:
            pred = self.model(x)
            loss = self.criterion(pred, y)
            self.losses_testing = np.append(self.losses_testing, loss.detach().item())          # log the loss on testing set
        return self.losses_testing 
    
    def optimize_ctrl(self, load_opt_name):
        """
        Fix the NN's weights; optimize the input ctrl_params to 
        obtain the P* that gives best fidelity
        """
        self.opt.load_state_dict(torch.load(load_opt_name))                                     # load the optimizer pt file
        def infidelity(ctl_params):
            """
            ctrl_params are aray with dim=(L,3)
            """
            # sum(Parallel(n_jobs=n_cores, verbose=0)(delayed(devi_sigma_O_T)(o,o,qubit_params,bath_params)  for  o in  _O_ ))
            return self.model(np.array(ctl_params ,dtype='float32').flatten())

        _opt_ = opt.minimize(fun = infidelity,x0= [np.pi, 0,0] * inputs.shape[1] , method ='Nelder-Mead',options={'maxiter': 1000})
        return  _opt_.x


#f = open(os.path.dirname(__file__)+"/simu_data_Gaussian_L=4_multiT_False.ds", 'rb') 
#data = pickle.load(f)
#f.close()
#inputs  = np.array(data["training_inputs"] ,dtype='float32')
#targets = np.array(data["training_targets"],dtype='float32') 
#print(torch.reshape(torch.from_numpy(inputs[:] ), (-1,12)).shape)
#print(torch.reshape(torch.from_numpy(targets[:] ), (-1,18)).shape)
#
#
#
#f = open(os.path.dirname(__file__)+"/simu_data_Gaussian_L=4_multiT_True.ds", 'rb') 
#data = pickle.load(f)
#f.close()
#inputs  = np.array(data["training_inputs"] ,dtype='float32')
#targets = np.array(data["training_targets"],dtype='float32')
#print(torch.from_numpy(inputs[:]).shape)
#print(torch.reshape(torch.from_numpy(targets[:] ), (-1,3*6*4)).shape)


############################################################################################################
#########        training the NN
############################################################################################################

if __name__ == '__main__':
    
    ################ [1] Gaussian control ####################

    # @ fixed time T
    f = open(os.path.dirname(__file__)+"/simu_data_Gaussian_L=4_multiT_False.ds", 'rb') 
    data = pickle.load(f)
    f.close()

    inputs  = np.array(data["training_inputs"] ,dtype='float32')
    targets = np.array(data["training_targets"],dtype='float32') 
    print("Gaussian_ctrl. The inputs shape is: ",inputs.shape, ". targets shape is: ", targets.shape)

    print('start training NN Gaussian ctrl:')
    my_training_G = QubitNeuralNetwork(inputs, targets)                   # Instantiate the class
    my_training_G.get_data_shape()                                        # print the data shape
    my_training_G.train()                                                 # training
    torch.save(my_training_G.opt.state_dict(), \
            join(dirname(abspath(__file__)), "optimizer_G_mulT=F_L=%d.pt"%(inputs.shape[1])))          # Save optimizer 
    
    #//// https://stackoverflow.com/questions/74626924/state-dict-error-for-testing-a-model-that-is-trained-by-loading-checkpoints
    my_training_G.test(join(dirname(abspath(__file__)), "optimizer_G_mulT=F_L=%d.pt"%(inputs.shape[1])))    # test
    print(my_training_G.losses_testing)


    # @ varying time
    f = open(os.path.dirname(__file__)+"/simu_data_Gaussian_L=4_multiT_True.ds", 'rb') 
    data = pickle.load(f)
    f.close()

    inputs  = np.array(data["training_inputs"] ,dtype='float32')
    targets = np.array(data["training_targets"],dtype='float32') 
    print("Gaussian_ctrl. The inputs shape is: ",inputs.shape, ". targets shape is: ", targets.shape)

    print('start training GRU Gaussian ctrl:')
    my_training_G = QubitRecurrentNetwork(inputs, targets)                # Instantiate the class
    my_training_G.get_data_shape()                                        # print the data shape
    my_training_G.train()                                                 # training
    torch.save(my_training_G.opt.state_dict(), \
            join(dirname(abspath(__file__)), "optimizer_G_mulT=T_L=%d.pt"%(inputs.shape[1])))          # Save optimizer 
    
    #//// https://stackoverflow.com/questions/74626924/state-dict-error-for-testing-a-model-that-is-trained-by-loading-checkpoints
    my_training_G.test(join(dirname(abspath(__file__)), "optimizer_G_mulT=T_L=%d.pt"%(inputs.shape[1])))    # test
    print(my_training_G.losses_testing)    







    ################  [2] Triangle control ####################
#    f = open(os.path.dirname(__file__)+"/simu_data_Triangle_L=1_multiT_False.ds", 'rb')
#    data = pickle.load(f)
#    f.close()
#
#    inputs  = np.array(data["training_inputs"] ,dtype='float32')
#    targets = np.array(data["training_targets"],dtype='float32')
#    print("Triangle_ctrl. The inputs shape is: ",inputs.shape, ". targets shape is: ", targets.shape) 

#    print('start training Triangle ctrl:')
#    my_training_T = QubitNeuralNetwork(inputs, targets)                   # Instantiate the class 
#    my_training_T.train()                                                 # training
#    torch.save(my_training_T.opt.state_dict(), \
#        join(dirname(abspath(__file__)), "optimizer_T_L=%d.pt"%(inputs.shape[1])))    # Save optimizer
#    my_training_T.test(join(dirname(abspath(__file__)), "optimizer_T_L=%d.pt"%(inputs.shape[1])))    # test
#    print(my_training_T.losses_testing)


