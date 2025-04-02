from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import os
import time
import numpy as np
import scipy as sp
import torch
from BA_curvature_util import get_hypersphere_volume, sample_n_ball_uniform
from BA_testing_functions import CostFunction
from BA_grid_TASC import calc_landscape_tasc, generate_grid_point_array, one_iteration_grid_TASC, determine_outliers_in_grid, print_array_summary
from classic_training import cost_func
from data import random_unitary_matrix
from landscapes import generate_data_points
from metrics import calc_scalar_curvature_for_function, calc_several_scalar_curvature_values
from multiprocessing import Pool

from qnns.qnn import get_qnn


class Engine(object):
    def __init__(self, function, r, grid_point_array):
        self.function = function
        self.r = r
        self.grid_point_array = grid_point_array
    def __call__(self, idx):
        point = []
        # generate point in grid array
        i=0
        for dimension in idx:
            point.append(self.grid_point_array[i][dimension])
            i += 1
        point = np.asarray(point)
        #print(f"Calculating for point {j}/{no_of_points}", point.tolist())
        # calculate tasc, tsc, mean absolute sc and mean sc values at this point
        idx, tasc, tsc, masc, msc = calc_several_scalar_curvature_values_parallel1(self.function, self.r, point,idx=idx)
        return idx, tasc, tsc, masc, msc


def calc_several_scalar_curvature_values_parallel1(function, r, c, idx, N=1000, absolute=True):
    '''
    Calculates the total (absolute) scalar curvature and mean (absolute) scalar curvature
    of a function within a hypersphere of radius r around a center point c.

    Args:
        function (array): function 
        r (float): radius of hypersphere, same in every dimension
        c (array): point within loss landscape, center of hypersphere, array with n entries (one for each dimension)
        sampling (String): sampling method, possible values: 
            "uniform" (uniformly random, Marsaglia method) (default)
        N (int): number of sample points, default: 1000 #TODO: andere Zahl?

    Returns:
        float: total absolute scalar curvature
        float: total scalar curvature
        float: mean absolute scalar curvature
        float: mean scalar curvature
    '''
    dimensions = len(c)
    # get sample points within hypersphere
    sample_points = sample_n_ball_uniform(dimensions, r, c, N) #TODO: wenn andere sampling Methoden implementiert wurden erweitern
    scalar_curvature_landscape = calc_scalar_curvature_for_function(function, sample_points)
    # calculate total (absolute) scalar curvature
    # get volume of hypersphere
    hypersphere_volume = get_hypersphere_volume(dimensions, r)
    # compute total absolute sc
    total_absolute_sc = np.sum(np.absolute(scalar_curvature_landscape))
    total_sc = np.sum(scalar_curvature_landscape)
    total_absolute_sc = total_absolute_sc * hypersphere_volume/N
    total_sc = total_sc * hypersphere_volume/N

    # calculate mean (aboslute) scalar curvature
    mean_sc = np.mean(scalar_curvature_landscape)
    mean_asc = np.mean(np.absolute(scalar_curvature_landscape))
    return idx, np.round(total_absolute_sc,3), np.round(total_sc,3), np.round(mean_asc,3), np.round(mean_sc,3)

def one_iteration_grid_TASC_parallel1(function, lower_left, stepsize, directory="", info="", iteration=0, N=3,plots=False):
    '''
        Calculates TASC of a function for every point on a grid and outputs corresponding information, such as outliers and plots.
    '''
    date = datetime.today().strftime('%Y-%m-%d')
    dim = len(lower_left)
    filename = f"results/preliminary_tests/entire_grid/QNN_cost_{dim}D_N={N}_{date}.txt"
    f = open(filename, "a")
    f.write(f"Info: "+info+"\n")
    #sys.stdout = open("plots/preliminary_tests/cost_function/third_test_2025-03-11/result.txt", 'w')
    label = "TASC"
    start = time.time()
    print("Generating grid point array...")
    points = generate_grid_point_array(stepsize,lower_left,N)
    print("Calculating TASC landscape...")
    tasc_landscape,tsc_landscape,masc_landscape,msc_landscape = calc_landscape_tasc_parallel1(function, points, r=stepsize)
    print("Number of Grid Points", N)
    #print("Grid Points", points.tolist())
    #print("TASC", tasc_landscape.tolist())
    f.write("TASC"+"\n")
    f.close()
    print_array_summary(tasc_landscape, filename)
    outlier_points, outlier_values = determine_outliers_in_grid(points, tasc_landscape)
    f=open(filename, "a")
    f.write("Outlier Points"+str(outlier_points.tolist())+"\n")
    f.write("Outlier TASC Values"+str(outlier_values.tolist())+"\n")

    f.write("TSC"+"\n")
    f.close()
    print_array_summary(tsc_landscape, filename)
    outlier_points, outlier_values = determine_outliers_in_grid(points, tsc_landscape)
    f=open(filename, "a")
    f.write("Outlier Points"+str(outlier_points.tolist())+"\n")
    f.write("Outlier TSC Values"+str(outlier_values.tolist())+"\n")

    f.write("MASC"+"\n")
    f.close()
    print_array_summary(masc_landscape, filename)
    outlier_points, outlier_values = determine_outliers_in_grid(points, masc_landscape)
    f=open(filename, "a")
    f.write("Outlier Points"+str(outlier_points.tolist())+"\n")
    f.write("Outlier MASC Values"+str(outlier_values.tolist())+"\n")

    f.write("MSC"+"\n")
    f.close()
    print_array_summary(msc_landscape, filename)
    outlier_points, outlier_values = determine_outliers_in_grid(points, msc_landscape)
    f=open(filename, "a")
    f.write("Outlier Points"+str(outlier_points.tolist())+"\n")
    f.write("Outlier MSC Values"+str(outlier_values.tolist())+"\n")

    elapsed_time = time.time()-start
    f.write(f"Time (minutes): {np.round(elapsed_time/60,3)}")
    f.close()

def calc_landscape_tasc_parallel1(function, grid_point_array, r=0):
    '''
        Parallelisiert: Punkte im Gitter
        Calculate total absolute scalar curvature of the given function 
        in a ball of radius r at every point given in the grid_point_array.

        Args:
            function (callable):
            grid_point_array (array): array of arrays of node values for grid for every dimension
            r (float): radius of ball, default: r=0 (r will be calculcated to equal stepsize of grid)

        Results:
            tasc_landscape (array): calculated tasc value at every point in the given grid
    '''
    if grid_point_array.shape[0] == 0:
        raise Exception("grid has to contain at least two points.")
    # calculate correct radius r if it was not given
    if r==0:
        r = grid_point_array[0][1]-grid_point_array[0][0] # r = stepsize
    dim = grid_point_array.shape[0]
    # generate tasc_landscape 
    N = len(grid_point_array[0])
    landscape_shape = []
    for _ in range(dim):
        landscape_shape.append(N)
    landscape_shape = tuple(landscape_shape)
    tasc_landscape = np.empty(landscape_shape)
    tsc_landscape = np.empty(landscape_shape)
    mean_asc_landscape = np.empty(landscape_shape)
    mean_sc_landscape = np.empty(landscape_shape)

    #concurrency starts here
    engine = Engine(function, r, grid_point_array)
    cpu_count = os.cpu_count()
    assert cpu_count is not None
    print(f"CPU Count: {cpu_count}")
    with ProcessPoolExecutor(max_workers=cpu_count) as exe:
        futures = [exe.submit(engine, idx) for idx,_ in np.ndenumerate(tasc_landscape)]            
        for future in as_completed(futures):
            # get the result for the next completed task
            idx, tasc, tsc, masc, msc = future.result()# blocks
            tasc_landscape[idx],tsc_landscape[idx],mean_asc_landscape[idx], mean_sc_landscape[idx] = tasc, tsc, masc, msc
    exe.shutdown(wait=True, cancel_futures=False)
    return tasc_landscape, tsc_landscape, mean_asc_landscape, mean_sc_landscape  

def single_cost_test():
    num_qubits = 2
    cost_func = CostFunction(num_qubits=num_qubits,s_rank=4, num_data_points=4, data_type=4)
    ll = np.zeros(num_qubits*3)
    N=5
    stepsize = 2*np.pi/(N-1)
    print("--------------------------------------------")
    print("NO CONCURRENCY")
    print("--------------------------------------------")
    #one_iteration_grid_TASC(cost_func, lower_left=ll, stepsize=stepsize, N=5)
    print("Time (hours) 50+")
    print("--------------------------------------------")
    print("WITH CONCURRENCY")
    print("--------------------------------------------")
    one_iteration_grid_TASC_parallel1(cost_func,lower_left=ll,stepsize=stepsize,N=N, info=f"N={N}, dim={num_qubits*3}, s_rank=4, num_data_points=4, data_type=4")

if __name__=="__main__":
    single_cost_test()