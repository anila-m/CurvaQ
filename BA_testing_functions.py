import numpy as np
import torch

from classic_training import cost_func
from data import random_unitary_matrix
from landscapes import generate_data_points
from metrics import calc_scalar_curvature_for_function
from qnns.cuda_qnn import CudaPennylane, CudaSimpleEnt
from qnns.qnn import get_qnn

class CostFunction:
    def __init__(self,*, num_qubits=1, s_rank=0, num_data_points=0, data_type=0,unitary=None,inputs=None):
        '''
            Args:
                unitary: numpy array of values (not torch.tensor)
                inputs: numpy array of values (not torch.tensor)
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

def cosine_2D(x):
    if len(x) != 2:
        raise Exception
    Z = (1/2) * (np.cos(2 * x[0]) + np.cos(2 * x[1]))
    return Z

def f(x):
    if x.shape[0] != 2:
        raise Exception("Input has to have length of 2.")
    return x[0]**2 - x[1]**2

def get_ASC_function(func):
    def absolute_scalar_curvature(x):
        points = [x.tolist()]
        result,_,_ = calc_scalar_curvature_for_function(func, points)
        return np.absolute(result[0])
    return absolute_scalar_curvature


def get_basic_3D_cost_function(s_rank=1, ndp=1,type_of_data =1):
    '''
        deprecated, but may be in use somewhere, so it's staying for now.
    '''
    num_qubits = 1
    num_layers = 1
    
    num_data_points = ndp

    schmidt_rank = s_rank
    qnn = get_qnn("CudaU2", list(range(num_qubits)), num_layers, device="cpu") #FP: qnn = CudaPennylane(num_wires=num_qubits, num_layers=num_layers, device="cpu") 
    unitary = torch.tensor(data=np.array(random_unitary_matrix(num_qubits)), dtype=torch.complex128, device="cpu")
    inputs = generate_data_points(type_of_data=type_of_data, schmidt_rank=schmidt_rank, num_data_points=num_data_points, U=unitary, num_qubits=num_qubits)
    dimensions = num_qubits * num_layers * 3
    x = inputs
    expected_output = torch.matmul(unitary, x)
    y_true = expected_output.conj()
    def cost_function(x_in):
        qnn.params = torch.tensor(x_in, dtype=torch.float64, requires_grad=True).reshape(qnn.params.shape)
        cost = cost_func(inputs, y_true, qnn, device="cpu") 
        return cost.item()
    return cost_function

def get_basic_6D_cost_function(s_rank=1, ndp=1,type_of_data =1):
    '''
        deprecated, but may be in use somewhere, so it's staying for now.
    '''
    num_qubits = 2
    num_layers = 1
    
    num_data_points = ndp

    schmidt_rank = s_rank
    qnn = get_qnn("CudaU2", list(range(num_qubits)), num_layers, device="cpu") #FP: qnn = CudaPennylane(num_wires=num_qubits, num_layers=num_layers, device="cpu") 
    unitary = torch.tensor(data=np.array(random_unitary_matrix(num_qubits)), dtype=torch.complex128, device="cpu")
    inputs = generate_data_points(type_of_data=type_of_data, schmidt_rank=schmidt_rank, num_data_points=num_data_points, U=unitary, num_qubits=num_qubits)
    dimensions = num_qubits * num_layers * 3
    x = inputs
    expected_output = torch.matmul(unitary, x)
    y_true = expected_output.conj()
    def cost_function(x_in):
        qnn.params = torch.tensor(x_in, dtype=torch.float64, requires_grad=True).reshape(qnn.params.shape)
        cost = cost_func(inputs, y_true, qnn, device="cpu") 
        return cost.item()
    return cost_function