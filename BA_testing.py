from datetime import datetime
import os
import sys
import time
from matplotlib import cm
import numpy as np
#import autograd.numpy as np
from flask import json
import matplotlib.pyplot as plt
from scipy.optimize import rosen, rosen_der, rosen_hess
import scipy as sp

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

def g(x):
    return 1

def vectorize(f):
    vf = np.vectorize(f,signature="(n) -> ()")

    def newfunc(*args, **kwargs):
        return vf(*args, **kwargs)[()]
    return newfunc

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
    c = np.array([1,1,1,1,1,1])

    num_qubits = 2
    num_layers = 1
    
    num_data_points = 1

    for s in range(1,5,1):
        schmidt_rank = s
        qnn = get_qnn("CudaU2", list(range(num_qubits)), num_layers, device="cpu") #FP: qnn = CudaPennylane(num_wires=num_qubits, num_layers=num_layers, device="cpu") 
        unitary = torch.tensor(data=np.array(random_unitary_matrix(num_qubits)), dtype=torch.complex128, device="cpu")
        inputs = generate_data_points(type_of_data=1, schmidt_rank=schmidt_rank, num_data_points=num_data_points, U=unitary, num_qubits=num_qubits)
        dimensions = num_qubits * num_layers * 3
        x = inputs
        expected_output = torch.matmul(unitary, x)
        y_true = expected_output.conj()
        def cost_function(x_in):
            qnn.params = torch.tensor(x_in, dtype=torch.float64, requires_grad=True).reshape(qnn.params.shape)
            cost = cost_func(inputs, y_true, qnn, device="cpu") 
            return cost.item()
        results = []
        times = []
        for _ in range(50):
            start = time.time()
            tasc = calc_total_absolute_scalar_curvature(cost_function,r,c,N=1000,absolute=True)
            elapsed_time = time.time()-start
            results.append(tasc)
            times.append(elapsed_time)
        print(f"TASC of cost function (dim={dimensions}), r = {r}, c = {c}, num_qubits = {num_qubits}, num_layers = {num_layers}, schmidt_rank = {schmidt_rank}, num_data_points = {num_data_points}")
        print("results (50 runs)", results)
        print("median", np.median(results))
        print("mean", np.mean(results))
        print("std", np.std(results))
        print("variance", np.var(results))
        print("times (50 runs)",times)
        print("average time needed (sec): ", np.mean(times))
    
def test_tasc(dim=3):
    r = 1
    c = np.ones(dim)
    results = []
    times = []
    for _ in range(1000):
        start = time.time()
        tasc = calc_total_absolute_scalar_curvature(rosen,r,c,N=1000,absolute=True)
        elapsed_time = time.time()-start
        results.append(tasc)
        times.append(elapsed_time)
    print(f"TASC of rosenbrock (dim={dim}), r = {r}, c = {c}, N=1000")
    print("")
    print("results (100 runs)", results)
    print("")
    print("median", np.median(results))
    print("mean", np.mean(results))
    print("std", np.std(results))
    print("variance", np.var(results))
    print("")
    print("average time needed (sec): ", np.mean(times))

def testing_rosenbrock_3D(dim=3,r=1,runs=1000,N=1000):
    c = np.ones(dim)
    date = datetime.today().strftime('%Y-%m-%d')
    results = {"info": f"Rosenbrock, {dim}D, N=1000, r = {r}, c = {c.tolist()}", "date": date}
    tasc_results = np.ndarray((runs,))
    for run in range(runs):
        tasc, sc_values, sample_points = calc_total_absolute_scalar_curvature(rosen_projection_to_2d, r, c,N=N)
        tasc_results[run] = tasc
        result = {"tasc": tasc, "sample points": sample_points.tolist(), "SC values": sc_values.tolist()}
        results[run] = result
    tasc_results = list(tasc_results)
    results["all tasc results"] = tasc_results
    summary = {"median": float(np.median(tasc_results)), "mean": float(np.mean(tasc_results)), "std": float(np.std(tasc_results)), "variance": float(np.var(tasc_results)), "max": float(np.max(tasc_results)), "max runs": int(np.argmax(tasc_results))}
    results["summary"] = summary
    # write results to json file
    directory = "results/preliminary_tests"
    
    file = open(f"{directory}/rosenbrock{dim}D_projection_N={N}_{date}.json", mode="w")
    json.dump(results, file, indent=4)

def plot_rosenbrock():
    '''
        plotting rosen([x,y,1]) (3D) in [0,2]x[0,2]
    '''
    x = np.linspace(0,2,100)
    X, Y = np.meshgrid(x,x)
    ax = plt.subplot(111, projection='3d')
    Z = np.ones_like(X)
    ax.plot_surface(X, Y, rosen([X, Y, Z]))
    ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='Rosenbrock($x_1$,$x_2$,1)')
    plt.title("Rosenbrock function for $x_3=1$")
    plt.savefig("plots/preliminary_tests/Rosenbrock_2Dprojection_4thView.png")

def get_max_N_values(list, N):
    idx = []
    values = []
    list_copy = list
    for i in range(N):
        max_value = np.max(list_copy)
        max_index = np.argmax(list_copy)
        list_copy[max_index] = -np.inf
        idx.append(max_index)
        values.append(max_value)
    return values, idx

def plot_rosenbrock_SC():
    file = "results/preliminary_tests/rosenbrock2D_projection_2025-02-12.json"
    with open(file) as f:
        d = json.load(f)
        result = d["822"]
        sc_values = np.asarray(result["SC values"])
        sample_points = np.asarray(result["sample points"])
        X = sample_points[:,0]
        Y = sample_points[:,1]
        tasc = float(result["tasc"])
        max_index = np.argmax(sc_values)
        max_sample_point = sample_points[max_index]
        min_index = np.argmin(sc_values)
        min_sample_point = sample_points[min_index]
        print("Median", np.median(sc_values), "Mean", float(np.mean(sc_values)), "STD", float(np.std(sc_values)), "Variance", float(np.var(sc_values)))
        print("Max SC", np.max(sc_values), "Max SC Index", max_index, "Max SC Sample Point", max_sample_point)
        print("Min SC", np.min(sc_values), "Min SC Index", min_index, "Min SC Sample Point", min_sample_point)
        ax = plt.subplot(111, projection='3d')
        ax.scatter(X, Y, sc_values, c=sc_values,cmap="viridis")
        ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='SC')
        plt.title(f"Scalar Curvature of Rosenbrock function for $x_3=1$. TASC = {tasc}")
        plt.savefig("plots/preliminary_tests/Rosenbrock_2Dprojection_SC.png")
        plt.close()
        del ax
        ax = plt.subplot()
        ax = plt.boxplot(sc_values)
        plt.title(f"Rosenbrock function for $x_3=1$: Scalar Curvature Boxplot.\nTASC = {tasc}")
        plt.savefig("plots/preliminary_tests/Rosenbrock_2Dprojection_SC_boxplot.png")

        max_values, max_indexes = get_max_N_values(sc_values,10)
        s1 = "SC"
        s2 = "Index"
        s3 ="Sample Point"
        print(f"{s1:<20}  {s2:<10}  {s3:<30}")
        for idx in range(len(max_indexes)):
            sc = max_values[idx]
            index = max_indexes[idx]
            sample_point = sample_points[index]
            print(f"{sc:<20}  {index:<10}  {np.array2string(sample_point):<30}")
    return 0

def rosen_projection_to_2d(x):
    y = np.append(x,1)
    return rosen(y)

def calc_SC(gradient_vector, point_hessian):
    '''
        Calculate Scalar Curvature given the first and seconder order derivates (gradient and hessian) at a certain point.
    '''
    beta = 1 / (1 + np.linalg.norm(gradient_vector) ** 2)
    left_term = beta * (
            np.trace(point_hessian) ** 2
            - np.trace(np.matmul(point_hessian, point_hessian))
        )
    right_inner = np.matmul(point_hessian, point_hessian) - np.trace(
            point_hessian
        ) * point_hessian
    # order of matmul with gradient does not matter
    right_term = (
            2
            * (beta**2)
            * (np.matmul(np.matmul(gradient_vector.T, right_inner), gradient_vector))
        )
    point_curv = left_term + right_term
    return point_curv

def test_rosenbrock_derivatives():
    '''
    Testing Rosenbrock (3D) derivatives at point [1,1,1] and calculate SC value
    '''
    x = np.ones(3)
    print("actual derivatives, calculated with rosen_der and rosen_hess (scipy) at point [1,1,1]")
    grad = rosen_der(x)
    hess = rosen_hess(x)
    print("gradient: ", grad)
    print("hessian: ", hess)
    print("SC: ", calc_SC(grad,hess))

    grad = sp.optimize.approx_fprime(x, rosen)
    hess = calc_hessian(rosen, x)
    print("approximations used to calculate SC values")
    print("gradient: ", grad)
    print("hessian: ", hess)
    print("SC: ", calc_SC(grad,hess))



def plot_rosenbrock_SC():
    grid_size = 100 # in total 100x100 points on regular grid
    x = np.linspace(0.99,1.01,grid_size)
    X,Y = np.meshgrid(x,x)
    points = np.ndarray((grid_size**2,2))
    points[:,0] = X.flatten()
    points[:,1] = Y.flatten()
    sc_values = calc_scalar_curvature_for_function(rosen_projection_to_2d,points)
    ax = plt.subplot(111, projection='3d')
    ax.plot_surface(X, Y, sc_values.reshape(X.shape), rstride=1, cstride=1, cmap=cm.viridis, linewidth=0.1)
    ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='SC')
    plt.title("Scalar Curvature for Rosenbrock function for $x_3=1$")
    plt.savefig("plots/preliminary_tests/Rosenbrock_2Dprojection_SC_grid_abitsmaller.png", dpi=500)



if __name__ == "__main__":
    r = 1
    c = np.array([1,2,1])
    #test_uniform_sampling()
    #print(plot_f())
    #print(calc_total_absolute_scalar_curvature(f,r,c,N=1000))
    #test_uniform_sampling_costFunc()
    #print(calc_total_absolute_scalar_curvature(rosen,r,c,N=1000))
    #testing_rosenbrock_3D(dim=2)
    #test_rosenbrock_derivatives()
    plot_rosenbrock_SC()

