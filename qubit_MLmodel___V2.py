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
from scipy.linalg import hadamard
from scipy.linalg import expm
import scipy.integrate as integrate
#import scipy.special.erf as erf
from joblib import Parallel, delayed
from numba import njit
import time

from torch.utils.data import TensorDataset, DataLoader

n_cores = 6


############################################################################################################
#########        Paley ordering Walsh digital bases 
############################################################################################################

def standard_Walsh(N):
    """
    Return standard (N*N) Walsh matrix W_N:
    """
    if N == 1:
        return np.array([[1]])
    else:                                   

        H = standard_Walsh(N // 2)
        return np.block([[H, H],\
                        [H, -H] ])

def paley_Walsh(N):
    """
    Return paley order Walsh matrix W_N_paley [(N*N)]
    """


    W = standard_Walsh(N)
    n = W.shape[0]
    indices = list(range(n))

    # Function to count sign changes in a row
    def sign_changes(row):
        return np.sum(row[:-1] != row[1:])
    
    # Sort indices based on sign changes in the corresponding rows of H
    indices.sort(key=lambda i: sign_changes(W[i]))
    
    return W[indices]

W_8_paley = paley_Walsh(8)  # A proper paley_order Walsh-8   
# print("Walsh-8", W_8_paley) #test pass
# print("Walsh-16",paley_Walsh(16))  #test pass

############################################################################################################
#########        Walsh tranformation on the ctrl params: should be feed to NNs 
############################################################################################################

from qubit_simulator__v2 import NoisyQubitSimulator
from qubit_simulator__v2 import pauli_operators

class ControlF1Haar(NoisyQubitSimulator):
    """
    Compute the F1_u(l,n) for windowed-Haar frames  phi(l,n) where 
    l: the window-index: 0<=l<L, 
    n: the Walsh-index:  0<=n<N#
    -----------------------------------------------------------------
    Return:   (L,N_, 3)-dim array // F1_u(l,n)
    -----------------------------------------------------------------
    """
    def __init__(self,T, L, C_params, C_shape, N_=8 ) -> None:
        """
        Inherit from qubit_simulator:
        self.T, self.L, self.C_params, self.C_shape, self.MM, self.K, self.tau,....
        ------------------------------------------------------------------------------ 
        C_params: a list of len=L with each element = [theta, (alpha, beta)] gives the pulse amplitude & direction

        N_=8: the cutoff Walsh basis functions -- only use all W_8 functions, in Paley order 
             [all beyond like W_16[n] with 8<n<=16 are HIGH-frequency and dropped ]
        ------------------------------------------------------------------------------     
        """
        super().__init__(T, L, C_params, C_shape) 
        self.N_ = N_                                                # The only new property in this subclass

    def haar_frames(self,n, t_l):
        """
        The **simple**-Haar frames phi_(n)(t_l) where   
        0<=n<  N#, and  
        0<=t_l<= T/L =tau 
        """
        if t_l < 0 or t_l > self.tau:
            raise ValueError(f"t should be in the range 0 <= x <= (tau){self.tau}")
        if n<0 or n> self.N_:
            raise ValueError(f"n should be in the range 0<=n<=N_{self.N_}")
        walsh_mat = W_8_paley if self.N_==8 else  paley_Walsh(self.N_)                   # obtain the W_N in paley order
        return walsh_mat[n, int(self.N_ * t_l//self.tau)]                                # Map W_N's n-th row & N*t_/tau col 

    def U_ctrl_continuous(self, l, t_l):
        """
        Control propergator:
        l:  l-th window 0<=l<L, 
        t_l: RELATIVE time in that window & 0<t_l<tau
        --------------------------------------------------------
        It is indirect way to calculate U_ctrl at ANT CONTINUOIUS time t in [0,T]
        ------------------------------------------------------------------------
        Return: U_ctrl(l,t_l)  // Equiv way as U_ctrl(t) for t in [0,T]
        """
        # give the integration of waveform_profile: h_t_integral:
        if    self.C_shape == "Gaussian":
            profile = lambda t: 0.5 - 0.5*np.cos(5 - (10 * t)/self.tau)                         # Mathematica calculated \int^t_0 h(t') dt'
        elif  self.C_shape == "Triangle":
            profile = lambda t: (2 * t**2 / self.tau**2) if t < 0.5 * self.tau \
                                else ((4 * self.tau - 2 * t) * t / self.tau**2)                 # Mathematica calculated \int^t_0 h(t') dt'    
        return expm(-1j * profile(t_l) * \
                           (self.n_x[l]*pauli_operators[1]+\
                            self.n_y[l]*pauli_operators[2]+\
                            self.n_z[l]*pauli_operators[3])
                    )  \
                 @ (self.U_ctrl_n(l-1) if l!=0 else np.identity(2))               # Ongoing window @ ALL finished windows         
    
    def y_u(self,u, l, t_l):
        """
        u:  (int)  1<=u<=3  pauli x,y,z
        Return: the switching function  y_u(t) = Tr[U0d @ sig_z @ U0 @ sig_u] 
                As U0 = `U_ctrl_continuous()`,  0<t_l<tau 
        """
        if u <= 0 or u > 3:
            raise ValueError(f"u should be in the range 1 <= x <=3 for 3-Pauli")
        U0 = self.U_ctrl_continuous(l,t_l)
        return np.trace(np.matrix(U0).getH() @ np.matrix(pauli_operators[3]) @\
                        np.matrix(U0)        @ np.matrix(pauli_operators[u]) ).real
    
    #@njit
    def F1_haar(self, l, n):
        """
        Return the F1_u(l,n) = int^T_0 [y_u(t) phi_{l,n}(t)]dt 
                             = int^{l*tau}_{(l-1)tau} [y_u(t) W_n(t)]dt
                                for u ={x,y,z}
               shape = (3)- array                 
        """
        integrand = lambda ttt, u: self.y_u(u, l, ttt) * self.haar_frames(n, ttt)
        return np.array([
                    integrate.quad(integrand, 0, self.tau, args=(1,))[0],
                    integrate.quad(integrand, 0, self.tau, args=(2,))[0],
                    integrate.quad(integrand, 0, self.tau, args=(3,))[0]   ])
    
    def output_Layer(self):
        """
        Return  all F1_u(l,n) with the shape l,n,u 
        -> (L, N_,3) array
        """
        #print((self.L)*(self.N_))
        results =Parallel(n_jobs=n_cores, verbose=0) \
                         (delayed(self.F1_haar)(l, n) for l in range(self.L) for n in range(self.N_)   )
        table = np.array(results).reshape(self.L, self.N_, 3)
        return table


class ControlF1Haar_GPU(NoisyQubitSimulator):
    def __init__(self, T, L, C_params_batch, C_shape, N_=8, device='cuda'):
        """
        Initialize with batched C_params.
        C_params_batch: a tensor of shape (batch_size, L, 3)
        """
        super().__init__(T, L, C_params_batch[0], C_shape)
        self.N_ = N_
        self.device = torch.device(device)
        self.C_params_batch = C_params_batch.to(self.device)  # Shape: (batch_size, L, 3)
        self.batch_size = C_params_batch.shape[0]

    def haar_frames(self, n, t_l):
        walsh_mat = W_8_paley if self.N_ == 8 else paley_Walsh(self.N_)
        return walsh_mat[:, n, int(self.N_ * t_l // self.tau)]  # Shape: (batch_size, ...)

    def U_ctrl_continuous(self, l, t_l):
        if self.C_shape == "Gaussian":
            profile = lambda t: 0.5 - 0.5 * torch.cos(5 - (10 * t) / self.tau)
        elif self.C_shape == "Triangle":
            profile = lambda t: (2 * t ** 2 / self.tau ** 2) if t < 0.5 * self.tau \
                else ((4 * self.tau - 2 * t) * t / self.tau ** 2)
        
        U = torch.matrix_exp(-1j * profile(t_l) * 
                          (self.C_params_batch[:, l, 0].unsqueeze(1) * pauli_operators[1] + 
                           self.C_params_batch[:, l, 1].unsqueeze(1) * pauli_operators[2] + 
                           self.C_params_batch[:, l, 2].unsqueeze(1) * pauli_operators[3]))
        return U @ (self.U_ctrl_n(l-1).unsqueeze(0) if l != 0 else torch.eye(2, device=self.device).unsqueeze(0))

    def y_u(self, u, l, t_l):
        U0 = self.U_ctrl_continuous(l, t_l)
        return torch.real(torch.einsum('bij,bjk,bkl->bil', torch.conj(U0.transpose(1, 2)), pauli_operators[3], U0, pauli_operators[u]))

    def F1_haar(self, l, n):
        def integrand(ttt, u):
            return self.y_u(u, l, ttt) * self.haar_frames(n, ttt)
        
        result = torch.stack([
            torch.trapz(integrand(self.tau, 1), dim=-1),
            torch.trapz(integrand(self.tau, 2), dim=-1),
            torch.trapz(integrand(self.tau, 3), dim=-1)
        ], dim=-1)
        return result

    def output_Layer(self):
        results = torch.zeros((self.batch_size, self.L, self.N_, 3), dtype=torch.float32, device=self.device)
        for l in range(self.L):
            for n in range(self.N_):
                results[:, l, n, :] = self.F1_haar(l, n)
        return results


# test this part in a simple case:

T = 10**-6
C_params= [ [0.6*np.pi, 0.8*np.pi, -0.5*np.pi],\
            [1.4*np.pi, 2.30*np.pi, 0.7*np.pi],\
            [0.6*np.pi, -0.8*np.pi, -2.5*np.pi],\
            [1.14*np.pi, 1.130*np.pi, 0.17*np.pi],
            [0.6*np.pi, 0.8*np.pi, -0.5*np.pi],\
            [1.4*np.pi, 2.30*np.pi, 0.7*np.pi],\
            [0.6*np.pi, -0.8*np.pi, -2.5*np.pi],\
            [1.14*np.pi, 1.130*np.pi, 0.17*np.pi]]         
L = len(C_params)          
from datetime import datetime 
print(datetime.now().strftime("%H:%M:%S"))
ctrl_F1_layer = ControlF1Haar(T=T, L=L, C_params= C_params,  C_shape="Gaussian", N_ = 8)
#ctrl_F1_layer.F1_haar(3, 5)
start_time = time.time()
print(ctrl_F1_layer.F1_haar(3, 5) )
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
print(datetime.now().strftime("%H:%M:%S"))


############################################################################################################
#########        build NN model 
############################################################################################################


class MyNN(nn.Module):
    """
    define simple NN model: there are '4' hidden layers
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
        return (self.layers((x)))
    

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
        self.nInputs    = np.prod(inputs.shape[1:])                                               # nInput based on input data
        self.nOutputs   = np.prod(targets.shape[1:])                                              # nOutput based on outout data
        self.model      = MyNN(nInput  = self.nInputs, 
                               nHidden = 24, 
                               nOutput = self.nOutputs)                                           # Construct a NN model
        self.opt        = torch.optim.SGD(self.model.parameters(), lr=1e-4)                       # set optimizer of the NN model
        self.criterion  = nn.MSELoss()                                                            # set loss_function of the NN model
        self.losses_training   = []    
        self.losses_testing    = []                                                                      
        self.Epochs     = 10000
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
                
    def train(self):
        """
        training the neural network
        """
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

        _opt_ = opt.minimize(fun = infidelity,x0= [np.pi, 0,0] * self.inputs.shape[1] , method ='Nelder-Mead',options={'maxiter': 1000})
        return  _opt_.x


############################################################################################################
#########       🔴 🔴 🔴 🔴  training the NN  🔴 🔴 🔴 🔴
############################################################################################################

"""
if __name__ == '__main__':
    
    # Define the file path
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "X_pulses_L4_simu_data.pkl")
    L=4
    # Load the variables from the file
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
        print(f"Variables loaded from {file_path}")


    ################ [1] Gaussian control ####################
    inputs = np.array(loaded_data['X_sequence'], dtype='float32')
    inputs = np.array([ControlF1Haar(T=T,                                           # Regularize the ctrl_input to be in Haar frames
                                    L=L, 
                                    C_params= each,  
                                    C_shape="Gaussian", 
                                    N_ = 8).output_Layer() for each in inputs ] ,dtype='float32')                                     
    targets =np.array(loaded_data['Gauss_multiT_msmt'] , dtype='float32')
    print("Gaussian_ctrl. The inputs shape is: ",inputs.shape, ". targets shape is: ", targets.shape)   

    print('start training Gaussian ctrl:')
    my_training_G = QubitNeuralNetwork(inputs, targets)                   # Instantiate the class
    my_training_G.train()                                                 # training
    torch.save(my_training_G.opt.state_dict(), \
            join(dirname(abspath(__file__)), "optimizer_G_L=%d.pt"%(inputs.shape[1])))          # Save optimizer 
    
    #//// https://stackoverflow.com/questions/74626924/state-dict-error-for-testing-a-model-that-is-trained-by-loading-checkpoints
    my_training_G.test(join(dirname(abspath(__file__)), "optimizer_G_L=%d.pt"%(inputs.shape[1])))    # test
    print(my_training_G.losses_testing)



    ################  [2] Triangle control ####################
    inputs =  np.array(loaded_data['X_sequence'], dtype='float32')
    inputs = np.array([ControlF1Haar(T=T,                                           # Regularize the ctrl_input to be in Haar frames
                                    L=L, 
                                    C_params= each,  
                                    C_shape="Triangle", 
                                    N_ = 8).output_Layer() for each in inputs ] ,dtype='float32')         
    targets = np.array(loaded_data['Trian_multiT_msmt'] , dtype='float32')
    print("Triangle_ctrl. The inputs shape is: ",inputs.shape, ". targets shape is: ", targets.shape)   

    print('start training Triangle ctrl:')
    my_training_T = QubitNeuralNetwork(inputs, targets)                   # Instantiate the class 
    my_training_T.train()                                                 # training
    torch.save(my_training_T.opt.state_dict(), \
        join(dirname(abspath(__file__)), "optimizer_T_L=%d.pt"%(inputs.shape[1])))    # Save optimizer
    my_training_T.test(join(dirname(abspath(__file__)), "optimizer_T_L=%d.pt"%(inputs.shape[1])))    # test
    print(my_training_T.losses_testing)


""" 

