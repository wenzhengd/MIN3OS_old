"""
This module implements a simulator for a noisy qubit in the rotating frame
the inputs are 
1. noise trajectories
2. control_pulses
3. Evoltion time T and window L  ...

Outputs are
E[O] for all O and for all rho_S
"""

# preample
import numpy as np
from functools import reduce
from itertools import product
from scipy.linalg import expm
from datetime import datetime

###############################################################################

pauli_operators = [ np.array([[1,0],[0,1]]), 
                    np.array([[0,1],[1,0]]), 
                    np.array([[0,-1j],[1j,0]]), 
                    np.array([[1,0],[0,-1]]) ]
np.random.seed(seed = 42)

def RTN_generator(T, gamma, g, MM=1000, K=1000):
    """
    RTN sampling: a zero mean  zero-frequency-center telegraph noise
    output:  A shape = (K * MM) noise sample, where row->trajec col->time 
    ---------------------------------------------------------------------------------------
    If we know that state is s at time t: z_s(t), then at t+dt, the flip to s' has probablity
	P_flip(t, t+dt) = 1- e^(-gamma dt)
    ---------------------------------------------------------------------------------------
    T         : The total evolution time 
    gamma     : flipping rate of RTN
    g         : coupling
    MM        : Time discretization 
    K         : total noise realization
    """
    trajectory_table = np.zeros((K,MM))     #shape = num_sample * num_time_chop
    for i in range(K):
        trajectory_table[i][0] = 1 if (np.random.uniform(0,1)>0.5) else -1 #  \pm 1 full-random zero-mean
        j=1
        while j<MM:
            trajectory_table[i][j] = 1 * trajectory_table[i][j-1] if ( np.exp(-gamma* T/MM)  > np.random.uniform(0, 1)) \
                else -1* trajectory_table[i][j-1]
            j+=1
    # now add cos modulation 
	#for i in range(K):
	#	phi =  np.random.uniform(0, 1)*2*np.pi
	#	for j in range(MM):
	#		trajectory_table[i][j] = trajectory_table[i][j] * np.cos(Omega * j * dt + phi)
    return g * trajectory_table

class NoisyQubitSimulator():
    """
    Class for simulating a noisy  qubit  in rotating frame, but OUTPUT in toggling frame !!!!!!!
    ----------------------------------------------------------
    it used the RTN_generator generated noise trajectories
    input includes T,L and control pulses
    output includes E[O] for all O and for all rho_S
    """
    def __init__(self, T, L, C_params, C_shape="Gaussian", MM=1000, K=1000, MultiTimeMeas =False):
        """
        T               : Evolution time
        L               : Total num of windows = pulse_number
        tau             : One window time duration
        C_params        : a list of len=L with each element = [theta, (alpha, beta)] gives the pulse amplitude & direction
        C_shape         : Waveforms: Gaussian or Triangle
        MM              : Total time pieces in [0,T]
        K               : total noise trajec number
        """
        self.T          = T                 # Evolution time
        self.L          = L                 # Total num of windows = pulse_number
        self.C_params   = C_params          # A list of L sublists, each=[theta, alpha, beta] gives rot_amplitude, n_x= cos(a)cos(b), n_y = cos(a)sin(b), n_z= sin(a)
        self.C_shape    = C_shape           # Waveforms: Gaussian or Triangle (a single string)
        self.MM         = MM                # Total time pieces in [0,T]
        self.K          = K                 # total noise trajec number
        self.tau        = T/L               # One window time duration
        self.dt         = T/MM              # duration of each small time_piece
        self.theta      = [x[0] for x in C_params]                                      # pulse amplitude                  
        self.alpha      = [x[1] for x in C_params]                                      # pulse direction_alpha
        self.beta       = [x[2] for x in C_params]                                      # pulse direction_beta
        self.n_x        = [np.cos(a)*np.cos(b) for a,b in zip(self.alpha,self.beta)]    # convert ctrl_direction angle to n_x
        self.n_y        = [np.cos(a)*np.sin(b) for a,b in zip(self.alpha,self.beta)]    # convert ctrl_direction angle to n_y
        self.n_z        = [np.sin(a)           for a,b in zip(self.alpha,self.beta)]    # convert ctrl_direction angle to n_z
        self.U_ctrl_T   = self.U_ctrl_T()                                               # the final control propagator 
        self.trajectory = np.load('RTN_traj_hash.npy')
        self.MultiTimeMeas = MultiTimeMeas                                              # Varing msmt time ?	
        
    def set_ctrl_shape(self):
        """
        set a list of len = MM/L (only for 1 window)  NORMALIZED waveforms based on C_shape
        TODO: 可以放到 init 中, 只需计算一次.
        TODO: 丢掉 for loop, 完全用 NumPy
        """
        if self.C_shape == "Gaussian":
            h_t = lambda t: (5.64189583548624/self.tau)* np.exp( -t**2/(0.1*self.tau)**2)  *(t<0.5*self.tau)*(t>-0.5*self.tau)      # Gaussian symmetric & normalized  waveform : the bandwith = T/L/10
        elif  self.C_shape == "Triangle":
            h_t = lambda t: (4/self.tau**2) *((t<=0)*t + (t>0)*(-t) + self.tau/2)  *(t<0.5*self.tau)*(t>-0.5*self.tau)          # triangle symmetric & normalized  waveform    
        ts = np.linspace(-0.5*self.tau,0.5*self.tau, int(self.MM/self.L), endpoint = False)
        return h_t(ts + 0.5* self.dt )     # Discrete time_step sampling in ONE window
    
    def U_ctrl_T(self):
        """
        Output: U_ctrl(T), the full ctrl_propagator
        """
        unitary = pauli_operators[0]
        for n in range(self.L):  # L: Total num of windows = pulse_number
            unitary =  self.su_2(self.C_params[n][0],[self.n_x[n], self.n_y[n], self.n_z[n]] ) @ unitary
        return  np.matrix(unitary)
    
    def U_ctrl_n(self,n):
        """
        input: window index: n= 0, 1, ..., L-1 
        return U_ctrl after window-n FINISHED, (L-1) gives U_ctrl_T
        """
        unitary = pauli_operators[0]
        for n in range(n+1):
            unitary =  self.su_2(self.C_params[n][0],[self.n_x[n], self.n_y[n], self.n_z[n]] ) @ unitary
        return  np.matrix(unitary)

    def set_total_hamiltonian(self):
        """
        Output: a array of dimension= (K,M) total Hamiltonians --- H(t) --- based on C_params & C_shape &noise       
        """
        # pulse 是给定的, 所以 H_ctrl 是一个 deterministic 的值
        # 两个 Loop: for i in range(4), for j in range(250)
        # 计算量不大, 但是可以优化
        h_1window = self.set_ctrl_shape()                                               # list of ctrl_waveforms in one window     (len = MM/L)
        H_ctrl =  np.array([self.theta[n]* (self.n_x[n]*pauli_operators[1]+\
                                            self.n_y[n]*pauli_operators[2]+\
                                            self.n_z[n]*pauli_operators[3])\
                    *h  for n in range(self.L) for h in h_1window ])                    # list of control_Hamiltonian for each discrete time points (len = MM)
        H_ctrl = np.tile(H_ctrl,(self.K,1,1,1))                                         # Repeat the control_Hamiltonian K times for K trajectories in noise_ensemble
        H_sb = np.einsum("km,ij->kmij", self.trajectory, pauli_operators[3])            # Dephasing error Hamiltonian
        # print(np.allclose(H_sb, H_sb1))
        return H_ctrl + H_sb                                                            # H_total(t) = H_ctrl(t) +  H_sb(t)
            
    def evolution(self):
        """
        time-order propagator of the QE joint-system
        --------------------------------------------------------
        Output: :  an size (=K) array of (2*2) matrices,the U(T) =Propagate{H(t)} for ALL noise realization
        """
        if self.MultiTimeMeas == False:
            """
            only measure at t= T:
            Returen an size (=K) array of (2*2) matrices,the U(T) =Propagate{H(t)} for ALL noise realization
            """
            evolve = lambda U,U_j: U_j @ U                                                                   # define a lambda function for calculating the propagator
            total_hamiltonians = self.set_total_hamiltonian()
            total_hamiltonians = np.swapaxes(total_hamiltonians, 0, 1)  # MM * K * 2 * 2
            U_total = reduce(evolve, expm(-1j*self.dt* total_hamiltonians))  # MM * (K*2*2)                  # calculate and accumalate all propagators till the final one, and repeat over all realizations
            # revised 6 np.matmul.reduce 有问题 (似乎还没实现, 或者不支持 non-commutative operations), [::-1] 可能不利于后续的矩阵乘法(内存顺序问题)
            # total_hamiltonians = np.swapaxes(total_hamiltonians, 0, 1)  # MM * K * 2 * 2
            # U_total = reduce(np.matmul, expm(-1j*self.dt* total_hamiltonians)[::-1,:,:])  # MM * (K*2*2)
            # U_total = np.matmul.reduce(expm(-1j*self.dt* total_hamiltonians)[::-1,:,:], axis=1)
            # print(U_total.shape)
            #self.U_err = np.average(U_errr_all, axis =1)                                                    # averaging over all realizations
        else:
            """
            measure multiple-time at window n <= L
            Returen an size (=K) array of (L *2*2) matrices,
            the U_n(T) =Propagate{H(t)} for ALL noise realization 
            """
            evolve = lambda U,U_j: U_j @ U                                                                   
            total_Hamiltonian_list = self.set_total_hamiltonian()
            U_stage= []#U_stage = np.zeros((self.K, self.L), dtype= object)
            for n in range(self.L):
                """
                the chooped propagator for t in window (n-1) to window n
                """
                # Old method:
                # U_stage.append( [reduce(evolve, [expm(-1j*self.dt* time_slice) for time_slice in trajec]) \
                #                  for trajec in total_Hamiltonian_list[:, int(n*self.MM/self.L):int((n+1)*self.MM/self.L), :, :]]   )
                # new method:
                hamiltonians_n = total_Hamiltonian_list[:, int(n*self.MM/self.L):int((n+1)*self.MM/self.L), :, :]  #take window snippet of full
                hamiltonians_n = np.swapaxes(hamiltonians_n, 0, 1)
                U = reduce(evolve, expm(-1j * self.dt * hamiltonians_n))
                U_stage.append(U)
            U_stage = np.moveaxis(np.array(U_stage) , 0, 1)                     # shuffle the 1st two axies such that the dim =(K,L,2,2)
            for n in range(1,self.L): 
                """
                the propagator for t in window 0 to window n
                """
                for k in range(self.K):
                    U_stage[k,n] =  U_stage[k,n] @  U_stage[k,n-1]              # pre-pend the already evoleved early window
            U_total = U_stage            

        return np.array(U_total,dtype='complex')

    def readout_T(self):
        """
        Output:  <O>|_S for all O and all rho_S in "Toggling" frame 
        ---------------------------------
        Notice: since the evolution is in rotating frame and the ML_module will be working fror toggling frame dynamics [perturbation theory],
        the Output/readout here should be in T-frame. 
        we want tilde{O} = U_0^+ @ O @ U_0  = pauli
        thus the R-frame O = U_0 @ pauli @ U_0^+ to make sure T-frame's O properly cycle over pauli 
        """
        if self.MultiTimeMeas == False:
            """
            single time
            """
            msmt_O =  [self.U_ctrl_T @ O @ (self.U_ctrl_T.getH()) for O in  pauli_operators[1:]]                                    # All R-frame observables to make T-frame tilde{O} = pauli
            msmt_S = 1/2*np.array([(pauli_operators[0]+pauli_operators[1]), (pauli_operators[0]-pauli_operators[1]), \
                                    (pauli_operators[0]+pauli_operators[2]), (pauli_operators[0]-pauli_operators[2]),\
                                    (pauli_operators[0]+pauli_operators[3]), (pauli_operators[0]-pauli_operators[3]) ])              # All initial states
            U_all = self.evolution()
            results = np.zeros((len(msmt_O), len(msmt_S)))
            # results1 = np.zeros((len(msmt_O), len(msmt_S)))
            for idx_O, O in enumerate(msmt_O):            
                for idx_S, S in enumerate(msmt_S):
                    # below: average over all trajectories 
                    # results[idx_O,idx_S] = np.average( [ np.trace(np.matrix(U) @ S @ np.matrix(U).getH() @ O) for U in U_all] )     # calculate E[O(T)]_rhoS in Rotating frame
                    # revised1: remove np.matrix, use np.conj(U.T)
                    # results[idx_O,idx_S] = np.average( [ np.trace(U @ S @ np.conj(U.T) @ O) for U in U_all] )     # calculate E[O(T)]_rhoS in Rotating frame
                    # revised2: 小数点十几位之后有所不同, 但是 allclose 仍然返回 True
                    # results[idx_O,idx_S] = np.average( [ np.trace(np.einsum("ij,jk,kl,lm->im", U, S, np.conj(U.T), O)) for U in U_all] )     # calculate E[O(T)]_rhoS in Rotating frame
                    # revised3: remove list comprehension
                    results[idx_O,idx_S] = np.average( [ np.trace(np.einsum("nij,jk,nkl,lm->nim", U_all, S, np.conj(np.swapaxes(U_all,1,2)), O), axis1=1,axis2=2).real] )     # calculate E[O(T)]_rhoS in Rotating frame
            # the results are rotating frame simultion , yet it corresponds to toggling frame results with stantard \tidle{O} = pauli
            # print(np.allclose(results, results1))
        else:
            """
            multiple time
            """
            msmt_S = 1/2*np.array([(pauli_operators[0]+pauli_operators[1]), (pauli_operators[0]-pauli_operators[1]), \
                               (pauli_operators[0]+pauli_operators[2]), (pauli_operators[0]-pauli_operators[2]),\
                               (pauli_operators[0]+pauli_operators[3]), (pauli_operators[0]-pauli_operators[3]) ])                  # All initial states
            U_all = self.evolution()
            results = np.zeros((3,  6 , int(self.L)))
            for n in range(self.L):
                msmt_O = [self.U_ctrl_n(n) @ O @ (self.U_ctrl_n(n).getH()) for O in  pauli_operators[1:]]                           # All R-frame observables to make T-frame tilde{O} = pauli
                for idx_O, O in enumerate(msmt_O):            
                    for idx_S, S in enumerate(msmt_S):     
                        # below: average over all trajectories
                        results[idx_O,idx_S,n] = np.average( [ np.trace(np.matrix(U) @ S @ np.matrix(U).getH() @ O).real \
                                                                                    for U in U_all[:,n ,:, :]] )                    # calculate E[O(t_n)]_rhoS in Rotating frame
                        # the results are rotating frame simultion , yet it corresponds to toggling frame results with stantard \tidle{O} = pauli        
        
        return results                                                                                                          # Toggling frame 

    def su_2(self,angle,direction):
        """
        SU2 rotation 
        TODO: 把括号里改写为矩阵乘法
        TODO: 进一步, 把 n angles, n direction 同时计算出来
        """
        return np.cos(angle)*pauli_operators[0] -1j*np.sin(angle)*\
            (direction[0]*pauli_operators[1]+direction[1]*pauli_operators[2]+direction[2]*pauli_operators[3])


        
if __name__ == '__main__':
    import time
    tic = time.time()
    #################################################
    # generate noise trajectories
    ###################################################
    T =10**-6
    MM = 1000
    K  = 1000

    trajectory_main = RTN_generator(T=T, gamma=10**4, g= 0* 10* 10**5, MM=MM, K=K)
    # NO NEED TO SAVE: this only a local test
    np.save('RTN_traj_hash.npy', np.array(trajectory_main))	
    tic = time.time()
    # print(f"Elapsed time: {toc - tic} seconds")
    #################################################
    # generate simulation data of noisy qubits
    ###################################################
    noisy_qubit = NoisyQubitSimulator(T=T, L=4, 
                                      C_params=[[0.6*np.pi, 0.8*np.pi, -0.5*np.pi],\
                                                [1.4*np.pi, 2.30*np.pi, 0.7*np.pi],\
                                                [0.6*np.pi, -0.8*np.pi, -2.5*np.pi],\
                                                [1.14*np.pi, 1.130*np.pi, 0.17*np.pi]],\
                                       C_shape="Gaussian", MM=MM, K=K, MultiTimeMeas= False)
    #print(noisy_qubit.readout_T())
    #toc = time.time()
    #print(f"Elapsed time: {toc - tic} seconds")
    print(datetime.now().strftime("%H:%M:%S"))
    print(noisy_qubit.readout_T())
    print(datetime.now().strftime("%H:%M:%S"))
