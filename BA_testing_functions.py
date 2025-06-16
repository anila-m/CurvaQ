import numpy as np
import torch
from scipy.optimize import rosen

from classic_training import cost_func
from data import random_unitary_matrix
from landscapes import generate_data_points
from metrics import calc_scalar_curvature_for_function
from qnns.cuda_qnn import CudaPennylane, CudaSimpleEnt
from qnns.qnn import get_qnn


class CostFunction:
    def __init__(self,*, num_qubits=1, s_rank=0, num_data_points=0, data_type=0,unitary=None,inputs=None):
        '''
            Initializes QNN cost function based on unitary or inputs (training data), if unitary is None it is generated randomly,
            if inputs is None, Training data is generated from given Schmidt rank (s_rank), number of samples (num_data_points) and data type (data_type)
            
            Args:
                num_qubits (int): number of qubits in QNN PQC, default: 1, 
                s_rank (int): Schmidt rank of training samples, between 1 and 2 for 1-qubit QNN, default: 0, 
                num_data_points (int): Number of training samples, between 1 and 4, default: 0, 
                data_type (int): data type of training samples, between 1 and 4, default: 0,
                unitary: numpy array of values (not torch.tensor), default: None
                inputs: numpy array of values (not torch.tensor), default: None
        '''
        num_layers = 1
        schmidt_rank = s_rank
        self.qnn = CudaPennylane(num_wires=num_qubits, num_layers=num_layers, device="cpu") 
        n = 2**num_qubits
        if unitary is None:
            self.unitary = torch.tensor(data=np.array(random_unitary_matrix(num_qubits)), dtype=torch.complex128, device="cpu") #TODO: auf ein unitary festlegen?
        else:
            self.unitary = torch.from_numpy(unitary.reshape(-1,n))
        
        if s_rank==0:
            self.inputs = torch.from_numpy(inputs.reshape(-1,n,n))
        else:
            if(s_rank not in range(1,3) or num_data_points not in range(1,5) or data_type not in range(1,5)):
                raise Exception("Wrong Schmidt Rank, number of data points or data type")
            self.inputs = generate_data_points(type_of_data=data_type, schmidt_rank=schmidt_rank, num_data_points=num_data_points, U=self.unitary, num_qubits=num_qubits)
        self.dimensions = num_qubits * num_layers * 3
        expected_output = torch.matmul(self.unitary, self.inputs)
        self.y_true = expected_output.conj()

    def __call__(self, x_in):
        self.qnn.params = torch.tensor(x_in, dtype=torch.float64, requires_grad=True).reshape(self.qnn.params.shape)
        cost = cost_func(self.inputs, self.y_true, self.qnn, device="cpu") 
        return cost.item()

k=2
def cosine_2D(x):
    '''
        Test function for thesis chapter "Curvature". f(x,y) = 1/(2^k)*(cos(2^k*x)+cos(2^k*y)), k=2
    '''
    if len(x) != 2:
        raise Exception
    Z = (1/2**k) * (np.cos(2**k * x[0]) + np.cos(2**k * x[1]))
    return Z

def rosen_projection_to_2d(x):
    '''
        Reduction of Rosenbrock function to 2D, by setting last input x_3 to 1. 
        Args:
            x (list): input values
        Returns:
            Rosen(x,1)
    '''
    y = np.append(x,1)
    return rosen(y)

def get_ASC_function(func, absolute=True):
    '''
        Function that computes absolute scalar curvature for func.
    '''
    def absolute_scalar_curvature(x):
        points = [x.tolist()]
        result,_,_ = calc_scalar_curvature_for_function(func, points)
        if absolute:
            return np.absolute(result[0])
        else:
            return result[0]
    return absolute_scalar_curvature

