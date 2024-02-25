
import numpy as np
from qubit_simulator import NoisyQubitSimulator
from multiprocessing import Pool
from joblib import Parallel, delayed
from itertools import product
import pickle
import os.path

n_cores = 6

T = 10**-6 
L = 4
num_diff_ctrls = 10000
pulse_sequences = [[[2*np.pi* np.random.rand() for k in range(3)] for j in range(L) ] for i in range(num_diff_ctrls)]  
MM=1000 

#print(pulse_sequences)
######################################################################
##    simulate noisy qubit with 'Gaussian' waveform control
######################################################################
def noisy_qubit_1(pulse_sequence):
    return NoisyQubitSimulator(T=T, L=L, C_params=pulse_sequence, C_shape="Gaussian", MM=MM ).readout_T()          # simulate noisy qubit with 'Gaussian' waveform control
#print(pulse_sequences[0])
#print(noisy_qubit_1(pulse_sequences[0]).readout_T())

#with Pool() as p:
#        simulation_results = p.starmap(noisy_qubit_1, pulse_sequences)

simulation_results = Parallel(n_jobs=n_cores, verbose=0)(delayed(noisy_qubit_1)(p_sequence) for p_sequence in pulse_sequences )

#print(simulation_results)
#np.save('simu_data_Gaussian.npy',simulation_results) # //// why it is not stored in the disk???

f = open("simu_data_Gaussian_L=%d.ds"%L, 'wb')
pickle.dump({ "training_inputs":pulse_sequences, "training_targets":simulation_results}, f, -1)
f.close()

######################################################################
##    simulate noisy qubit with 'Triangle' waveform control
######################################################################
def noisy_qubit_2(pulse_sequence):
    return NoisyQubitSimulator(T=T, L=L, C_params=pulse_sequence, C_shape="Triangle", MM=MM ).readout_T()           # simulate noisy qubit with 'Triangle' waveform control

#with Pool() as p:
#        simulation_results = p.starmap(noisy_qubit_2, pulse_sequences)
simulation_results = Parallel(n_jobs=n_cores, verbose=0)(delayed(noisy_qubit_2)(p_sequence) for p_sequence in pulse_sequences )

#np.save('simu_data_Triangle.npy',simulation_results)

f = open("simu_data_Triangle_L=%d.ds"%L, 'wb')
pickle.dump({ "training_inputs":pulse_sequences, "training_targets":simulation_results}, f, -1)
f.close()

