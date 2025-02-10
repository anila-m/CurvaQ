import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import rosen

from metrics import *
from BA_curvature_util import *

step_size = 0.4



def test_volume():
    for dim in range(0,10):
        print(dim,get_hypersphere_volume(dim,2))

def test_uniform_sampling():
    r = 5
    c = [1,1]
    n = len(c)
    #print(norm)
    print("----------------")
    samples = sample_n_ball_uniform(n,r,c,1000)
    # check if samples are within n-ball
    x = samples[:,0]
    y = samples[:,1]
    
    circ = plt.Circle(c,r,color="r",fill=False)
    plt.figure(figsize=(6,6))
    fig = plt.gcf()
    ax = fig.gca()
    ax.add_patch(circ)
    plt.scatter(x,y)
    plt.title(f"Uniformly sampled points, radius {r} and center {c}")
    plt.show()

    return 0

def f(x):
    return np.sum(x, axis=1)
def h(x):
    return np.prod(x, axis=1)

def plot_f():
    # x and y axis
    X,Y,Z = [], [], []
    for x in np.arange(-5,5,0.1):
        for y in np.arange(-5,5,0.1):
            X.append(x)
            Y.append(y)
            Z.append(f([x,y]))
    
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot3D(X, Y, Z, color ='green')
    ax.set_title('wireframe geeks for geeks')
    plt.show()

def test_uniform_sampling_costFunc():
    r = 1
    c = np.array([0,0,0])

    num_qubits = 1
    num_layers = 1
    schmidt_rank = 1
    num_data_points = 1
    qnn = get_qnn("CudaU2", list(range(num_qubits)), num_layers, device="cpu") #FP: qnn = CudaPennylane(num_wires=num_qubits, num_layers=num_layers, device="cpu") 
    unitary = torch.tensor(data=np.array(random_unitary_matrix(num_qubits)), dtype=torch.complex128, device="cpu")
    inputs = generate_data_points(type_of_data=1, schmidt_rank=schmidt_rank, num_data_points=num_data_points, U=unitary, num_qubits=num_qubits)
    dimensions = num_qubits * num_layers * 3
    x = inputs
    expected_output = torch.matmul(unitary, x)
    y_true = expected_output.conj()
    def cost_function(x_in):
        print(x_in.shape)
        qnn.params = torch.tensor(x_in, dtype=torch.float64, requires_grad=True).reshape(qnn.params.shape)
        cost = cost_func(inputs, y_true, qnn, device="cpu") 
        return cost.item()
    start = time.time()
    tasc = calc_total_absolute_scalar_curvature(cost_function,r,c,N=1000,absolute=True)
    elapsed_time = time.time()-start
    print(tasc, elapsed_time)
    
def test_tasc():
    results = []
    for n in range(10):
        start = time.time()
        tasc = calc_total_absolute_scalar_curvature(h,r,c,N=2000,absolute=False)
        elapsed_time = time.time()-start
        results.append([tasc, elapsed_time])
    print(results)
        

if __name__ == "__main__":
    r = 1
    c = np.array([0,0])
    #test_uniform_sampling()
    #print(plot_f())
    #print(calc_total_absolute_scalar_curvature(f,r,c,N=1000))
    test_uniform_sampling_costFunc()


