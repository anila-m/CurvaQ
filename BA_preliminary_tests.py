from datetime import datetime
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
from BA_testing_functions import CostFunction, rosen_projection_to_2d
from BA_experiment_resources import unitary, unitary_3D
from BA_analysis_util import make_boxplot

step_size = 0.4

# Several preliminary tests, that were used during initial phase of thesis (i.e. implementation)

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
    '''
        Test TASC calculation with Naive MC Integration for Rosenbrock function (default 3D)
        Args:
            dim (int): dimension of Rosenbrock function, default: 3
    '''
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
    '''
        Test TASC calculation with Naive MC Integration for Rosenbrock function (default 3D) in a n-ball with center (1,...,1) and radius 1
        using 1000 sample points and over 1000 runs.
        Results saved in json file.

        Args:
            dim (int): dimension of Rosenbrock function, default: 3
            r (float): radius, default: 1
            runs (int): number of repititions, default: 1000
            N (int): number of sample points, default: 1000
    '''
    c = np.ones(dim)
    date = datetime.today().strftime('%Y-%m-%d')
    results = {"info": f"Rosenbrock, {dim}D, N=1000, r = {r}, c = {c.tolist()}", "date": date}
    tasc_results = np.ndarray((runs,))
    for run in range(runs):
        tasc, sc_values, sample_points = calc_total_absolute_scalar_curvature(rosen, r, c,N=N)
        tasc_results[run] = tasc
        result = {"tasc": tasc, "sample points": sample_points.tolist(), "SC values": sc_values.tolist()}
        results[run] = result
    tasc_results = list(tasc_results)
    results["all tasc results"] = tasc_results
    summary = {"median": float(np.median(tasc_results)), "mean": float(np.mean(tasc_results)), "std": float(np.std(tasc_results)), "variance": float(np.var(tasc_results)), "max": float(np.max(tasc_results)), "max runs": int(np.argmax(tasc_results))}
    results["summary"] = summary
    # write results to json file
    directory = "results/preliminary_tests"
    
    file = open(f"{directory}/rosenbrock{dim}D_N={N}_{date}.json", mode="w")
    json.dump(results, file, indent=4)

def get_max_N_values(list, N):
    '''
        Helper function: get maximum N values in list
        Args:
            list (list): list of values
            N (int)
        Returns:
            values: list of maximum N values
            idx: list of corresponding indices
    '''
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


def analyse_rosenbrock_gradient():
    '''
        Analyze norm of gradient and hessian of 2D rosenbrock reduction around minimum (1,1,1).
    '''
    #r_values = [1, 0.1, 0.01, 0.001, 0.0001]
    r_values = [0.01]
    c= np.asarray([1,1,1,1,1])
    print(f"Analyzing gradient and hessian norm around {c} for 2D Rosenbrock")
    print(f"number of sample points per radius: 100000")
    print("----------------------------")
    maximums = {}
    for r in r_values:
        sample_points = sample_n_ball_uniform(n=5, r=r, c=c, N=100)
        gradients = []
        hessians = []
        for point in sample_points:
            gradients.append(sp.optimize.approx_fprime(point, rosen))
            hessians.append(calc_hessian(rosen, point))
        gradients = np.asarray(gradients)
        hessians = np.asarray(hessians)
        print(f"Radius {r}")
        grad_norm = np.linalg.norm(gradients, axis=1)
        hess_norm = np.linalg.norm(hessians, axis=(1,2))
        print(hessians.shape)
        print(hess_norm.shape)
        print(sample_points.shape)
        max_grad = np.max(grad_norm)
        max_hess = np.max(hess_norm)
        print("Gradient Norm")
        print(f"Median: {np.median(grad_norm)}, Mean: {np.mean(grad_norm)}, Minimum: {np.min(grad_norm)}, Maximum: {max_grad}")
        print("Hessian Norm")
        print(f"Median: {np.median(hess_norm)}, Mean: {np.mean(hess_norm)}, Minimum: {np.min(hess_norm)}, Maximum: {max_hess}")
        print("----------------------------")
        maximums[r] = [max_grad, max_hess]
    return maximums


def calc_SC(gradient_vector, point_hessian):
    '''
        Helper function: Calculate Scalar Curvature given the first and seconder order derivates (gradient and hessian) at a certain point.
        Args:
            gradient_vector (array): gradient at point
            point_hessian (array): hessian at point
        Returns:
            point_curv (float): scalar curvature at point
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


def plot_rosenbrock_fun_and_SC(left=0.0, right=2.0):
    '''
        Creates one plot: Z-axis of Rosenbrock(x_1,x_2,1) function values and 
        colormap of corresponding Scalar Curvature (SC) values.
        Needed for Thesis (Chapter: Implementation).

        Args:
            left (float): leftmost point in grid
            right (float): rightmost point in grid 
    '''
    grid_size = 100 # in total 100x100 points on regular grid
    x = np.linspace(left,right,grid_size)
    X,Y = np.meshgrid(x,x)
    points = np.ndarray((grid_size**2,2))
    points[:,0] = X.flatten()
    points[:,1] = Y.flatten()
    sc_values,_,_ = calc_scalar_curvature_for_function(rosen_projection_to_2d,points)
    fun_values =[]
    for idx in range(points.shape[0]):
        fun_values.append(rosen_projection_to_2d(points[idx]))
    fun_values = np.asarray(fun_values)

    # Contour plot of SC
    fig1 = plt.figure(figsize=(8,6))
    ax1 = fig1.add_subplot(111)
    c = ax1.contourf(Y,X,sc_values.reshape(X.shape), cmap=cm.viridis)
    cbar = fig1.colorbar(c)
    cbar.ax.tick_params(labelsize=14)
    ax1.set_title("Scalar Curvature", fontsize=16)
    ax1.set_xlabel(xlabel='$x_1$', fontsize=14)
    ax1.set_ylabel(ylabel='$x_2$', fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    #ax1.tick_params(axis='x', labelrotation=20)
    plt.savefig("plots/preliminary_tests/test_functions/rosen2D_SC_contour.pdf", format="pdf", bbox_inches='tight', pad_inches=0.3)
    plt.close()

    
    fun_values =[]
    for idx in range(points.shape[0]):
        fun_values.append(rosen_projection_to_2d(points[idx]))
    fun_values = np.asarray(fun_values)

    # Surface plot of function values
    fig2 = plt.figure(figsize=(8,6))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot_surface(X,Y, fun_values.reshape(X.shape), rstride=1, cstride=1, cmap=cm.viridis, linewidth=0.1)
    ax2.set_xlabel(xlabel='$x_1$', fontsize=12)
    ax2.set_ylabel(ylabel='$x_2$', fontsize=12)
    ax2.set_zlabel(zlabel='$rosen(x_1,x_2,1)$', fontsize=12)
    #ax2.set_title("$rosen(x_1,x_2,1)$", fontsize=14)
    ax2.tick_params(axis='x', labelrotation=20)
    plt.savefig("plots/preliminary_tests/test_functions/rosen2D_fun.pdf", format="pdf", bbox_inches='tight', pad_inches=0.3)

def test_different_N_for_function(function, c, r, function_name, function_title):
    '''
        Calculates TASC value for function in n-ball of radius r with center c 100 times for each N, 
        tests different number sample points N = 1000, 10000, 50000.
        Makes a Boxplot of TASC values for each N.
        Args:
            function (callable): function who's SC is to be integrated
            c (list): center of n-ball, 
            r (float): radius of n-ball, 
            function_name (String): name of function for file (boxplot), 
            function_title (String): name of function for title of plot
    '''
    Ns = [500, 1000, 10000, 50000]
    tasc_values = dict.fromkeys(Ns)
    times = dict.fromkeys(Ns)
    for N in Ns:
        tasc_values[N] = []
        times[N] = []
        for i in range(50):
            start_time = time.time()
            tasc,_,_ = calc_total_absolute_scalar_curvature(function, r, c, N=N)
            elapsed_time = time.time()-start_time
            times[N].append(elapsed_time)
            tasc_values[N].append(tasc)
    # print average computational time needed for each number of samples N + relative difference
    print("Average Computational time needed and relative difference for function", function_name)
    print("N = 1000", np.round(np.mean(times[1000]),3))
    for i in range(1,4):
        rel_diff = (np.mean(times[Ns[i]])-np.mean(times[Ns[i-1]]))/np.mean(times[Ns[i-1]])*100
        print(f"N = {Ns[i]}", np.round(np.mean(times[Ns[i]]),3), f"+{np.round(rel_diff,1)}%")
    print(times)
    # print TASC information for each number of samples N + relative difference
    print("TASC mean, median, std, variance for function", function_name)
    for i in range(4):
        N = Ns[i]
        print(f"N = {N}", np.round(np.mean(tasc_values[N]),3), np.round(np.median(tasc_values[N]),3), np.round(np.std(tasc_values[N]),3), np.round(np.var(tasc_values[N]),3))
    print(tasc_values)
    # Boxplot of TASC values per number of samples N        
    file_name = f"{function_names[i]}_boxplot_per_N"
    make_boxplot(
            tasc_values[i],
            y_label="TASC",
            x_label="Number of samples",
            path=f"/preliminary_tests/N",
            filename=file_name,
            title=function_titles[i],
            no_outliers=False
        )
    make_boxplot(
            tasc_values[i],
            y_label="TASC",
            x_label="Number of samples",
            path=f"/preliminary_tests/N",
            filename=file_name,
            title=function_titles[i],
            no_outliers=True
        )

def test_different_N():
    '''
        Test Naive MC Integration of 4 test functions (3D QNN, 6D QNN, Rosenbrock 3D, Rosenbrock reduction to 2D)
        for different number of sample points (N = 500, 1000, 10000, 50000).
    '''
    r = 1
    dimensions = [2,3,3,6]
    u_3D = np.asarray(unitary_3D, dtype=complex)
    u_6D = np.asarray(unitary, dtype=complex)
    cost3D = CostFunction(num_qubits=1, unitary=u_3D, s_rank=1, num_data_points=1, data_type=1)
    cost6D = CostFunction(num_qubits=2, unitary=u_6D, s_rank=1, num_data_points=1, data_type=1)
    functions= [rosen_projection_to_2d, rosen, cost3D, cost6D]
    function_names = ["rosen2Dred", "rosen3D", "cost3D", "cost6D"]
    function_titles = ["$rosen(x_1, x_2, 1)$", "$rosen(x_1, x_2, x_3)$", "3D QNN Cost Function", "6D QNN Cost Function"]
    for i in range(4):
        c = np.ones(dimensions[i])
        function = functions[i]
        f_name = function_names[i]
        f_title = function_titles[i]
        print(f_name, datetime.now())
        test_different_N_for_function(function, c, r, f_name, f_title)


if __name__ == "__main__":
    # Create boxplots for differen values of N from past experiments
    function_names = ["rosen2Dred", "rosen3D", "cost3D", "cost6D"]
    function_titles = ["$rosen(x_1, x_2, 1)$", "$rosen(x_1, x_2, x_3)$", "3D QNN Cost Function", "6D QNN Cost Function"]
    
    tasc_values = {
        0:{500: [0.171, 0.006, 0.036, 0.042, 0.014, 0.013, 0.008, 0.014, 0.011, 0.014, 1.123, 0.796, 0.879, 0.254, 0.014, 0.009, 0.035, 0.028, 0.253, 0.008, 0.533, 0.164, 0.032, 0.022, 0.023, 0.03, 0.097, 0.009, 0.47, 0.121, 0.042, 0.073, 0.016, 0.004, 0.07, 0.02, 0.047, 0.12, 0.009, 0.008, 2.515, 0.002, 0.716, 0.021, 0.083, 0.02, 0.049, 0.011, 0.011, 0.04], 1000: [0.278, 0.02, 0.017, 0.115, 0.019, 0.054, 0.012, 10.953, 0.011, 0.206, 0.012, 0.036, 0.103, 0.08, 0.027, 0.137, 0.235, 0.018, 0.076, 0.086, 0.032, 0.145, 0.035, 0.013, 0.046, 0.027, 0.028, 3.977, 0.016, 0.033, 0.098, 0.047, 0.674, 0.01, 0.011, 0.011, 0.008, 0.112, 0.061, 0.114, 0.009, 0.014, 0.03, 1.163, 0.623, 0.211, 0.93, 1.635, 0.9, 0.073], 10000: [0.79, 0.275, 232.734, 0.178, 0.086, 0.091, 5.741, 0.48, 0.103, 8.494, 2.99, 0.189, 1.175, 4.101, 4.879, 0.028, 3.427, 14.386, 0.335, 0.726, 68.139, 3.254, 0.081, 0.13, 0.12, 0.257, 0.061, 0.263, 2.771, 2.669, 0.107, 0.148, 5.177, 0.218, 0.456, 2.363, 0.696, 0.041, 201.474, 0.412, 1.317, 96.45, 3.693, 1.041, 0.223, 0.16, 0.219, 0.216, 1.554, 14.583], 50000: [1.044, 0.737, 1.775, 1.932, 0.179, 9.582, 6.586, 1.17, 0.717, 0.385, 6.757, 1.54, 0.215, 0.654, 35.534, 8.135, 3.103, 0.385, 1.949, 0.469, 2.142, 0.396, 4.084, 24.201, 8.294, 2.766, 1.991, 0.444, 0.384, 21.092, 2.74, 11.161, 2.564, 38.06, 2.86, 0.245, 1.941, 0.384, 54.329, 0.686, 8.721, 4.599, 9.499, 0.25, 3.65, 6.46, 7.372, 2.146, 0.196, 1.451]},
        1:{500: [8.977, 10.665, 12.168, 10.559, 12.429, 11.156, 13.065, 12.185, 11.033, 7.822, 11.009, 10.668, 9.833, 9.295, 11.9, 8.833, 12.049, 11.27, 11.016, 10.018, 10.71, 11.608, 9.575, 10.201, 10.584, 10.972, 9.315, 11.565, 13.547, 10.725, 9.635, 10.144, 12.413, 12.375, 9.397, 11.273, 8.579, 11.459, 10.252, 9.704, 9.475, 12.034, 10.999, 10.72, 10.386, 8.982, 11.617, 11.257, 12.871, 10.647], 1000: [10.989, 10.326, 10.695, 9.859, 10.086, 9.911, 10.601, 9.074, 10.55, 10.567, 9.844, 11.988, 10.141, 31.7, 10.149, 11.076, 11.513, 10.42, 11.448, 15.678, 9.936, 11.346, 17.786, 10.254, 10.799, 9.951, 9.725, 12.409, 11.119, 10.613, 10.661, 10.957, 18.243, 9.481, 10.576, 11.326, 10.174, 9.51, 10.697, 10.408, 12.511, 9.968, 10.201, 125.813, 9.953, 10.311, 12.814, 10.66, 9.711, 11.056], 10000: [11.059, 240.383, 20.948, 17.288, 10.883, 22.8, 10.636, 50.635, 83.251, 10.641, 10.883, 10.628, 10.796, 13.148, 10.495, 10.959, 10.205, 12.284, 36.276, 11.076, 10.685, 11.097, 10.799, 10.391, 10.685, 16.655, 11.513, 11.297, 28.841, 11.157, 10.9, 50.048, 13.544, 113.047, 11.565, 10.516, 10.39, 10.425, 11.278, 40.645, 31.895, 21.148, 22.225, 10.963, 11.999, 17.057, 11.69, 17.851, 10.841, 10.832], 50000: [11.022, 10.59, 10.871, 11.222, 14.544, 18.703, 10.875, 13.126, 11.956, 20.946, 29.998, 14.657, 12.162, 47.641, 13.694, 11.466, 10.852, 76.217, 38.961, 29.273, 12.684, 11.152, 11.396, 10.748, 19.123, 20.252, 11.387, 11.67, 12.029, 71.492, 11.455, 11.772, 24.406, 12.769, 13.146, 18.64, 13.718, 22.534, 34.618, 10.992, 11.441, 12.046, 12.871, 14.906, 21.35, 11.159, 10.801, 34.387, 17.436, 84.847]},
        2:{500: [55.623, 57.653, 61.584, 55.1, 52.154, 57.651, 52.318, 57.345, 58.356, 58.411, 57.707, 57.13, 62.977, 57.172, 62.148, 57.918, 56.296, 52.49, 55.761, 61.054, 58.271, 63.916, 61.59, 56.036, 62.234, 57.111, 58.842, 51.689, 57.802, 56.148, 63.389, 59.389, 60.74, 53.635, 54.852, 62.747, 58.334, 56.035, 59.563, 60.056, 57.512, 57.484, 59.74, 59.859, 60.995, 59.787, 61.079, 60.008, 57.688, 59.465], 1000: [61.785, 60.266, 60.667, 59.514, 59.203, 58.31, 60.614, 57.972, 58.574, 58.573, 62.88, 60.161, 60.999, 58.106, 56.809, 60.222, 58.611, 57.407, 57.504, 61.932, 63.698, 60.864, 61.951, 55.502, 62.04, 59.206, 58.6, 56.473, 59.369, 55.001, 59.25, 59.479, 58.078, 62.845, 56.237, 60.899, 59.617, 57.421, 60.941, 59.954, 59.552, 57.386, 57.89, 58.636, 60.618, 59.599, 58.517, 58.113, 56.96, 60.391], 10000: [58.195, 59.326, 59.128, 58.421, 59.061, 58.338, 57.927, 58.487, 57.997, 58.75, 58.556, 59.297, 58.922, 58.338, 60.161, 59.775, 58.805, 59.121, 58.394, 59.058, 59.156, 58.32, 59.305, 58.365, 58.285, 59.049, 59.501, 59.823, 59.243, 59.0, 58.965, 58.561, 57.56, 58.034, 60.143, 58.934, 58.651, 59.1, 58.364, 59.178, 59.166, 58.839, 58.444, 58.643, 58.042, 58.272, 58.935, 58.348, 58.386, 58.529], 50000: [58.812, 58.398, 59.092, 58.957, 59.019, 58.61, 59.04, 58.44, 58.705, 58.899, 58.996, 58.546, 58.482, 58.8, 58.717, 59.032, 59.254, 59.296, 58.522, 58.536, 59.05, 58.676, 59.335, 59.286, 59.137, 58.723, 58.723, 58.496, 59.097, 58.737, 58.704, 58.581, 58.608, 59.419, 58.728, 58.812, 58.264, 58.878, 58.7, 59.319, 59.057, 59.288, 58.789, 58.502, 58.542, 58.855, 58.969, 58.73, 59.05, 59.066]},
        3:{500: [300.45, 287.847, 294.323, 294.748, 327.081, 296.348, 305.79, 309.057, 293.595, 311.134, 307.332, 308.245, 295.61, 298.096, 316.725, 289.677, 307.435, 306.127, 291.144, 313.263, 295.402, 304.841, 296.625, 303.765, 276.488, 300.002, 319.912, 307.752, 315.764, 307.346, 300.456, 298.014, 302.354, 315.109, 315.458, 303.495, 290.134, 303.519, 285.489, 284.871, 291.265, 308.917, 306.458, 319.764, 297.037, 311.106, 303.849, 313.964, 301.448, 296.817], 1000: [298.183, 297.825, 309.095, 304.795, 305.145, 296.501, 293.731, 297.863, 303.323, 298.838, 302.567, 307.737, 298.343, 295.702, 299.64, 309.045, 305.866, 296.269, 306.187, 306.394, 290.619, 296.58, 306.462, 299.012, 301.906, 297.593, 297.294, 307.985, 317.662, 324.275, 309.692, 316.059, 299.745, 291.472, 305.616, 307.65, 318.179, 310.376, 297.839, 299.937, 309.328, 317.579, 311.271, 308.528, 309.114, 300.676, 308.843, 292.182, 301.498, 305.671], 10000: [303.192, 302.424, 302.826, 301.196, 308.073, 306.327, 303.396, 307.381, 308.411, 304.503, 304.273, 303.466, 304.207, 305.58, 305.929, 304.594, 307.698, 305.91, 307.689, 301.058, 303.577, 303.848, 305.173, 303.05, 306.743, 305.242, 306.172, 306.918, 307.864, 305.777, 304.626, 303.41, 305.608, 307.059, 302.596, 301.962, 303.221, 303.234, 303.397, 299.242, 305.421, 308.336, 307.458, 303.999, 305.969, 301.067, 305.471, 303.282, 303.688, 301.438], 50000: [306.274, 305.268, 303.059, 304.983, 304.628, 303.095, 304.783, 303.612, 304.046, 304.395, 305.361, 304.302, 304.191, 304.136, 304.44, 306.179, 304.238, 304.543, 306.519, 303.627, 304.741, 303.449, 304.922, 303.93, 303.602, 304.979, 304.029, 305.312, 304.058, 303.349, 302.555, 303.032, 304.195, 305.162, 304.588, 304.62, 304.238, 304.6, 303.721, 303.545, 303.254, 303.81, 303.975, 303.971, 306.371, 305.923, 303.962, 304.111, 303.602, 305.051]}
    }

    for i in range(4):
        file_name = f"{function_names[i]}_boxplot_per_N"
        make_boxplot(
            tasc_values[i],
            y_label="TASC",
            x_label="Number of samples",
            path=f"/preliminary_tests/N",
            filename=file_name,
            title=function_titles[i],
        )
        make_boxplot(
            tasc_values[i],
            y_label="TASC",
            x_label="Number of samples",
            path=f"/preliminary_tests/N",
            filename=file_name,
            title=function_titles[i],
            no_outliers=True
        )


    

