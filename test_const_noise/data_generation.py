"""
author: wenzheng dong
----------------------
This module execute the MC simulation of NoisyQubitSimulator in qubit_simulator.py
to generate training data
==========
"""
import numpy as np
from qubit_simulator import NoisyQubitSimulator
from multiprocessing import Pool
from joblib import Parallel, delayed
from itertools import product
import pickle
import os.path
from os.path import abspath, dirname, join
import sys 
from datetime import datetime

n_cores = 6

T = 10**-6 
L = 4
MM=100 
num_diff_ctrls = 1000
np.random.seed(seed = 42)
pulse_sequences = [[[2*np.pi* np.random.rand() for k in range(3)] 
                    for j in range(L) ] for i in range(num_diff_ctrls)]  

#print(np.array(pulse_sequences).shape)
print(datetime.now().strftime("%H:%M:%S"))

######################################################################
######################################################################
##    simulate noisy qubit with 'Gaussian' waveform control
######################################################################
######################################################################

# @ fixed-time
def noisy_qubit_1(pulse_sequence):
    return NoisyQubitSimulator(T=T, L=L, C_params=pulse_sequence, C_shape="Gaussian", MM=MM, K=2 ,\
                               MultiTimeMeas =False ).readout_T()          # simulate noisy qubit with 'Gaussian' waveform control

simulation_results = Parallel(n_jobs=n_cores, verbose=0)(delayed(noisy_qubit_1)(p_sequence) for p_sequence in pulse_sequences )

#f = open("simu_data_Gaussian_L=%d.ds"%L, 'wb')
f = open(join(dirname(abspath(__file__)), "simu_data_Gaussian_L=%d_multiT_False.ds"%L), 'wb')
pickle.dump({ "training_inputs":pulse_sequences, "training_targets":simulation_results}, f, -1)
f.close()


# @ multiple-time
def noisy_qubit_1(pulse_sequence):
    return NoisyQubitSimulator(T=T, L=L, C_params=pulse_sequence, C_shape="Gaussian", MM=MM, K=2 ,\
                               MultiTimeMeas =True ).readout_T()          # simulate noisy qubit with 'Gaussian' waveform control

simulation_results = Parallel(n_jobs=n_cores, verbose=0)(delayed(noisy_qubit_1)(p_sequence) for p_sequence in pulse_sequences )

#f = open("simu_data_Gaussian_L=%d.ds"%L, 'wb')
f = open(join(dirname(abspath(__file__)), "simu_data_Gaussian_L=%d_multiT_True.ds"%L), 'wb')
pickle.dump({ "training_inputs":pulse_sequences, "training_targets":simulation_results}, f, -1)
f.close()




######################################################################
######################################################################
##    simulate noisy qubit with 'Triangle' waveform control
######################################################################
######################################################################

# @ fixed-time
def noisy_qubit_2(pulse_sequence):
    return NoisyQubitSimulator(T=T, L=L, C_params=pulse_sequence, C_shape="Triangle", MM=MM, K=2, \
                               MultiTimeMeas =False ).readout_T()           # simulate noisy qubit with 'Triangle' waveform control

simulation_results = Parallel(n_jobs=n_cores, verbose=0)(delayed(noisy_qubit_2)(p_sequence) for p_sequence in pulse_sequences )

#f = open("simu_data_Triangle_L=%d.ds"%L, 'wb')
f = open(join(dirname(abspath(__file__)), "simu_data_Triangle_L=%d_multiT_False.ds"%L), 'wb')
pickle.dump({ "training_inputs":pulse_sequences, "training_targets":simulation_results}, f, -1)
f.close()

# @ multiple-time
def noisy_qubit_2(pulse_sequence):
    return NoisyQubitSimulator(T=T, L=L, C_params=pulse_sequence, C_shape="Triangle", MM=MM, K=2, \
                               MultiTimeMeas =True ).readout_T()           # simulate noisy qubit with 'Triangle' waveform control

simulation_results = Parallel(n_jobs=n_cores, verbose=0)(delayed(noisy_qubit_2)(p_sequence) for p_sequence in pulse_sequences )

#f = open("simu_data_Triangle_L=%d.ds"%L, 'wb')
f = open(join(dirname(abspath(__file__)), "simu_data_Triangle_L=%d_multiT_True.ds"%L), 'wb')
pickle.dump({ "training_inputs":pulse_sequences, "training_targets":simulation_results}, f, -1)
f.close()

print(datetime.now().strftime("%H:%M:%S"))