from datetime import datetime
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import rosen
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import preprocessing
from scipy import stats

from BA_testing_functions import get_basic_6D_cost_function, get_basic_3D_cost_function
from metrics import *
from BA_testing import rosen_projection_to_2d

#z_score_threshold = 3

def generate_grid_point_array(stepsize, lower_left, N):
    '''
    Generates points in a regular grid in a hypercube, where lower_left is the lower left corner 
    of the hypercube and N is the number of nodes in each direction. 

    Args:
        stepsize (float):
        lower_left (array):
        N (int): number of points in each dimension

    Returns:
        grid_points (array): n-dimensional array, contains all points in grid
    '''
    dim = len(lower_left)
    x = [] 
    for i in range(dim):
        x_start = lower_left[i]
        x_end = x_start + (N-1)*stepsize
        x.append(np.linspace(x_start, x_end, N))

    return np.asarray(x)

def calc_landscape_tasc(function, grid_point_array, r=0):
    '''
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
    no_of_points = N**dim
    j = 1
    for idx, _ in np.ndenumerate(tasc_landscape):
        point = []
        # generate point in grid array
        i=0
        for dimension in idx:
            point.append(grid_point_array[i][dimension])
            i += 1
        point = np.asarray(point)
        #print(f"Calculating for point {j}/{no_of_points}", point.tolist())
        # calculate tasc, tsc, mean absolute sc and mean sc values at this point
        tasc_landscape[idx],tsc_landscape[idx],mean_asc_landscape[idx], mean_sc_landscape[idx] = calc_several_scalar_curvature_values(function, r, point)
        j += 1
    return tasc_landscape, tsc_landscape, mean_asc_landscape, mean_sc_landscape

def plot_2D_surface(points, values, label, title, file_name):
    '''
        Surface plot of values on a 2D grid (defined by points). 

        Args:
            points (array): 2xn array, that defines the points on the grid in each dimension
            values (array): nxn array, values corresponding to each point on the grid
            label (String): z-label in plot
            title (String): title for plot
            file_name (String): file name
    '''
    X,Y = np.meshgrid(points[0,:], points[1,:])
    ax = plt.subplot(111, projection='3d')
    ax.plot_surface(Y, X, values, rstride=1, cstride=1, cmap=cm.viridis, linewidth=0.1) #switch Y and X, since that's how our TASC value landscapes are computed
    ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel=label)
    plt.title(title)
    plt.savefig(f"plots/preliminary_tests/{file_name}.png", dpi=500)
    plt.close()

def plot_2D_overlapping_circles(points, values, label, title, file_name):
    '''
        Surface plot of values on a 2D grid (defined by points). 

        Args:
            points (array): 2xn array, that defines the points on the grid in each dimension
            values (array): nxn array, values corresponding to each point on the grid
            label (String): z-label in plot
            title (String): title for plot
            file_name (String): file name
    '''
    X,Y = np.meshgrid(points[0,:], points[1,:])
    df = pd.DataFrame({'X': Y.flatten(), #switching Y and X since that's how our value landscapes are computed
                   'Y':X.flatten(), 
                   'Z':values.flatten()})
    
    # get the Colour
    x              = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled       = min_max_scaler.fit_transform(x)
    df_S           = pd.DataFrame(x_scaled)
    c1             = df['Z']
    c2             = df_S[2]
    colors         = [cm.viridis(color) for color in c2]

    # Plot circles
    plt.figure()
    plt.grid()
    ax = plt.gca()
    ax.set_axisbelow(True)
    for a, b, color in zip(df['X'], df['Y'], colors):
        circle = plt.Circle((a, 
                            b), 
                            1, # Size
                            color=color, 
                            fill=True,
                            alpha=0.5)
        ax.add_artist(circle)

    plt.xlim([np.min(points[0]),np.max(points[0])])
    plt.ylim([np.min(points[1]),np.max(points[1])])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    ax.set_aspect(1.0)

    sc = plt.scatter(df['X'], df['Y'], s=0, c=c1, cmap='viridis', facecolors='none')

    cbar = plt.colorbar(sc)
    cbar.set_label(label, labelpad=10)
    plt.title(title)
    plt.savefig(f"plots/preliminary_tests/{file_name}.png", dpi=500)
    plt.close()

def plot_2D_scatter(points, values, label, title, file_name):
    '''
        Surface plot of values on a 2D grid (defined by points). 

        Args:
            points (array): 2xn array, that defines the points on the grid in each dimension
            values (array): nxn array, values corresponding to each point on the grid
            label (String): z-label in plot
            title (String): title for plot
            file_name (String): file name
    '''
    X,Y = np.meshgrid(points[0,:], points[1,:])
    df = pd.DataFrame({'X': Y.flatten(), #switching Y and X since that's how our value landscapes are computed
                   'Y':X.flatten(), 
                   'Z':values.flatten()})
    
    # get the Colour
    x              = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled       = min_max_scaler.fit_transform(x)
    df_S           = pd.DataFrame(x_scaled)
    c1             = df['Z']
    c2             = df_S[2]
    colors         = [cm.viridis(color) for color in c2]

    # Plot circles
    plt.figure()
    plt.grid()
    ax = plt.gca()
    ax.set_axisbelow(True)

    plt.xlim([np.min(points[0])-0.5,np.max(points[0])+0.5])
    plt.ylim([np.min(points[1])-0.5,np.max(points[1])+0.5])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    ax.set_aspect(1.0)

    sc = plt.scatter(df['X'], df['Y'], s=30, c=c1, cmap='viridis', facecolors='none')

    cbar = plt.colorbar(sc)
    cbar.set_label(label, labelpad=10)
    plt.title(title)
    plt.savefig(f"plots/preliminary_tests/{file_name}.png", dpi=500)
    plt.close()

def test_rosenbrock_grid():
    c = np.asarray([-5,-5])
    N=11
    points = generate_grid_point_array(1,c,N)
    print(points)
    date = datetime.today().strftime('%Y-%m-%d')
    results = {"date": date, "info": "Rosenbrock reduction to 2D: rosenbrock(x,y,1)", "grid points": points.tolist()}
    for i in range(20):
        start = time.time()
        tasc_landscape, tsc_landscape, mean_asc_landscape, mean_sc_landscape = calc_landscape_tasc(rosen_projection_to_2d, points)
        run_results = {"tasc": tasc_landscape.tolist(), "tsc": tsc_landscape.tolist(), "mean asc": mean_asc_landscape.tolist(), "mean sc": mean_sc_landscape.tolist()}
        results[i] = run_results
        elapsed_time = time.time()-start
        print("elapsed_time", elapsed_time)
    
    # write results to json file
    directory = "results/preliminary_tests/entire_grid"
    os.makedirs(directory, exist_ok=True)
    file = open(f"{directory}/rosenbrock2D_reduction_grid_N={N}_{date}.json", mode="w")
    json.dump(results, file, indent=4)

def determine_outliers_in_grid(points, values,z_score_threshold=3):
    '''
        Returns a list of points where the values (specified in values) is an outlier.
        Outliers are calculated using the Z-Score.

        Args:
            points (array): 2xn array, that defines the points on the grid in each dimension
            values (array): nxn array, values corresponding to each point on the grid
        
        Return:
            outlier_points (array): points where outlier occurs
            outlier_values (array): list of outliers associated with outlier_points
    '''
    # calculate Z-Score
    Z = stats.zscore(values.flatten())
    Z = Z.reshape(values.shape)
    # determine outliers (|Z-Score| > threshold) (typically: threshold=3)
    outlier_values = values[np.absolute(Z)>z_score_threshold]
    outlier_indices = np.nonzero(np.absolute(Z)>z_score_threshold)
    dim = len(outlier_indices)
    outlier_points = []
    for i in range(len(outlier_indices[0])):
        point = []
        for j in range(dim):
            point.append(points[j][outlier_indices[j][i]])
        outlier_points.append(point)
    outlier_points = np.asarray(outlier_points)
    return outlier_points, outlier_values

#TODO: weiter implementieren??
def determine_new_grid_from_cluster(cluster_points, N):
    '''
        Determines a new lower left point in a grid and a stepsize, such that all points inside
        the cluster are also in a grid, which has N nodes in each dimension

        Args:
            cluster_points (array): list of points inside the cluster (kXdim, where k is the number of points
                inside the cluster and dim is the number of dimensions)
            N (int): number of nodes in each dimension of the resulting grid
        
        Returns:

    '''
    if(cluster_points.shape[0]==0):
        raise Exception("Cluster of points cannot be empty")
    if(N<2):
        raise Exception("Grid has to consist of at least 2 points in each dimension. (N>=2)")
    dim = len(cluster_points[0])

def cost_func_grid_TASC():
    c = np.array([0,0,0])
    N=6
    points = generate_grid_point_array(1,c,N)
    num_qubits = 1
    num_layers = 1
    num_data_points = 1
    schmidt_rank = 1
    cost_function = get_basic_3D_cost_function()
    # compute TASC, TSC, Mean SC and Mean ASC values
    date = datetime.today().strftime('%Y-%m-%d')
    results = {"date": date, "info": f"QNN Cost Function (3D): num_qubits = {1}, num_layers = {1}, num_data_points = {1}, Schmidt = {1},", "grid points": points.tolist()}
    for i in range(1):
        start = time.time()
        tasc_landscape, tsc_landscape, mean_asc_landscape, mean_sc_landscape = calc_landscape_tasc(cost_function, points)
        run_results = {"tasc": tasc_landscape.tolist(), "tsc": tsc_landscape.tolist(), "mean asc": mean_asc_landscape.tolist(), "mean sc": mean_sc_landscape.tolist()}
        results[i] = run_results
        elapsed_time = time.time()-start
        print("elapsed_time", elapsed_time)
    
    # write results to json file
    directory = "results/preliminary_tests/entire_grid"
    os.makedirs(directory, exist_ok=True)
    file = open(f"{directory}/QNN_cost_3D_grid_N={N}_{date}.json", mode="w")
    json.dump(results, file, indent=4)


def plot_all_tasc_grids_from_run0():
    '''
        Plots all TASC, TSC, Mean ASC, Mean SC values from 
        "/results/preliminary_tests/entire_grid/rosenbrock2D_reduction_grid_N=11_2025-03-02.json"
        for Rosenbrock function (2D grid).
    '''
    labels = ["TASC", "TSC", "Mean ASC", "Mean SC"]

    tasc_values = np.asarray([
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.008,
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.006,
                    8.225,
                    0.172,
                    0.008,
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    11.457,
                    10.889,
                    0.142,
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.005,
                    35.346,
                    0.136,
                    0.005,
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.002,
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ]
            ])
    tsc_values = np.asarray([
            [
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0
            ],
            [
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0
            ],
            [
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                0.0,
                0.0,
                -0.0,
                -0.0
            ],
            [
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                0.008,
                0.0,
                0.0,
                -0.0,
                -0.0
            ],
            [
                -0.0,
                -0.0,
                0.0,
                0.0,
                0.005,
                -6.598,
                0.167,
                0.008,
                0.0,
                -0.0,
                -0.0
            ],
            [
                -0.0,
                -0.0,
                -0.0,
                0.0,
                -11.371,
                -5.813,
                0.05,
                -0.0,
                -0.0,
                -0.0,
                -0.0
            ],
            [
                -0.0,
                -0.0,
                -0.0,
                0.0,
                0.004,
                33.856,
                0.123,
                0.005,
                -0.0,
                -0.0,
                -0.0
            ],
            [
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                0.002,
                0.0,
                0.0,
                -0.0,
                -0.0
            ],
            [
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                0.0,
                -0.0,
                -0.0,
                -0.0
            ],
            [
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0
            ],
            [
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0
            ]
        ])
    masc_values = np.asarray([
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.003,
                0.0,
                0.0,
                0.0,
                0.0
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.002,
                2.618,
                0.055,
                0.003,
                0.0,
                0.0,
                0.0
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                3.647,
                3.466,
                0.045,
                0.0,
                0.0,
                0.0,
                0.0
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.002,
                11.251,
                0.043,
                0.002,
                0.0,
                0.0,
                0.0
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.001,
                0.0,
                0.0,
                0.0,
                0.0
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0
            ]
        ])
    msc_values = np.asarray([
            [
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0
            ],
            [
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0
            ],
            [
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                0.0,
                0.0,
                -0.0,
                -0.0
            ],
            [
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                0.003,
                0.0,
                0.0,
                -0.0,
                -0.0
            ],
            [
                -0.0,
                -0.0,
                0.0,
                0.0,
                0.002,
                -2.1,
                0.053,
                0.003,
                0.0,
                -0.0,
                -0.0
            ],
            [
                -0.0,
                -0.0,
                -0.0,
                0.0,
                -3.62,
                -1.85,
                0.016,
                -0.0,
                -0.0,
                -0.0,
                -0.0
            ],
            [
                -0.0,
                -0.0,
                -0.0,
                0.0,
                0.001,
                10.777,
                0.039,
                0.002,
                -0.0,
                -0.0,
                -0.0
            ],
            [
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                0.001,
                0.0,
                0.0,
                -0.0,
                -0.0
            ],
            [
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                0.0,
                -0.0,
                -0.0,
                -0.0
            ],
            [
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0
            ],
            [
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0
            ]
        ])
    points = np.asarray([
            [
                -5.0,
                -4.0,
                -3.0,
                -2.0,
                -1.0,
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0
            ],
            [
                -5.0,
                -4.0,
                -3.0,
                -2.0,
                -1.0,
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0
            ]
        ])
        
    all_values = [tasc_values, tsc_values, masc_values, msc_values]
    for i in range(len(labels)):
        label = labels[i]
        title = f"{label} for Rosenbrock function for $x_3=1$"
        file_name = f"{label}_grid_rosenbrock_run0"
        file_name2 = f"{label}_grid_circles_rosenbrock_run0"
        file_name3 = f"{label}_grid_scatter_rosenbrock_run0"
        values = all_values[i]
        plot_2D_surface(points, values, label, title, file_name)
        plot_2D_overlapping_circles(points, values, label, title, file_name2)
        plot_2D_scatter(points, values, label, title, file_name3)
        print("Outliers for ", label)
        a, b = determine_outliers_in_grid(points, values)
        print("Points", a)
        print("Values", b)

def plot_all_QNN_tasc_grids():
    '''
        Plots all TASC, TSC, Mean ASC, Mean SC values from 
        results/preliminary_tests/entire_grid/QNN_cost_3D_grid_N=6_2025-03-09.json
        for Rosenbrock function (2D grid).
    '''
    labels = ["TASC", "TSC", "Mean ASC", "Mean SC"]

    tasc_values = np.asarray([
            [
                [
                    39.282,
                    38.783,
                    36.369,
                    39.876,
                    38.035,
                    38.315
                ],
                [
                    29.167,
                    26.162,
                    27.751,
                    26.866,
                    26.022,
                    26.117
                ],
                [
                    21.915,
                    22.941,
                    19.692,
                    20.725,
                    21.416,
                    20.544
                ],
                [
                    23.891,
                    24.427,
                    24.477,
                    24.389,
                    24.518,
                    24.476
                ],
                [
                    29.067,
                    28.637,
                    29.729,
                    30.1,
                    27.677,
                    29.179
                ],
                [
                    31.936,
                    34.452,
                    33.376,
                    34.198,
                    31.807,
                    33.631
                ]
            ],
            [
                [
                    34.257,
                    34.216,
                    35.002,
                    35.544,
                    33.62,
                    34.055
                ],
                [
                    49.036,
                    49.808,
                    49.986,
                    48.092,
                    51.858,
                    49.386
                ],
                [
                    48.18,
                    46.549,
                    46.36,
                    47.064,
                    46.05,
                    48.25
                ],
                [
                    32.551,
                    33.738,
                    34.77,
                    32.622,
                    32.915,
                    32.346
                ],
                [
                    19.566,
                    18.898,
                    20.473,
                    19.323,
                    18.901,
                    20.368
                ],
                [
                    12.894,
                    13.044,
                    13.85,
                    12.993,
                    13.275,
                    13.984
                ]
            ],
            [
                [
                    28.156,
                    28.881,
                    30.29,
                    26.188,
                    29.442,
                    27.99
                ],
                [
                    54.467,
                    55.307,
                    55.136,
                    56.284,
                    54.855,
                    57.77
                ],
                [
                    67.734,
                    70.136,
                    70.389,
                    67.941,
                    69.202,
                    69.155
                ],
                [
                    46.099,
                    45.879,
                    46.685,
                    47.168,
                    46.934,
                    45.205
                ],
                [
                    16.264,
                    16.114,
                    16.831,
                    16.685,
                    15.925,
                    16.609
                ],
                [
                    13.321,
                    13.655,
                    13.725,
                    14.562,
                    14.081,
                    13.693
                ]
            ],
            [
                [
                    30.179,
                    30.347,
                    30.653,
                    29.272,
                    31.208,
                    29.963
                ],
                [
                    39.001,
                    42.263,
                    41.943,
                    44.268,
                    41.149,
                    42.977
                ],
                [
                    54.543,
                    52.837,
                    58.094,
                    54.216,
                    48.953,
                    52.216
                ],
                [
                    40.28,
                    44.789,
                    40.054,
                    43.123,
                    45.455,
                    41.362
                ],
                [
                    34.367,
                    30.051,
                    29.859,
                    32.477,
                    32.821,
                    36.831
                ],
                [
                    44.721,
                    43.901,
                    46.115,
                    43.808,
                    49.126,
                    44.66
                ]
            ],
            [
                [
                    34.228,
                    34.353,
                    34.504,
                    35.897,
                    32.697,
                    34.896
                ],
                [
                    22.823,
                    24.069,
                    24.131,
                    25.317,
                    24.512,
                    24.001
                ],
                [
                    19.94,
                    21.293,
                    17.901,
                    19.999,
                    20.089,
                    20.157
                ],
                [
                    37.369,
                    36.479,
                    33.531,
                    33.565,
                    33.842,
                    36.37
                ],
                [
                    88.458,
                    90.964,
                    86.379,
                    85.163,
                    95.251,
                    87.428
                ],
                [
                    102.776,
                    109.138,
                    109.446,
                    106.071,
                    112.005,
                    111.073
                ]
            ],
            [
                [
                    33.61,
                    32.68,
                    31.836,
                    29.466,
                    31.027,
                    30.964
                ],
                [
                    11.898,
                    12.919,
                    11.901,
                    12.67,
                    12.19,
                    11.958
                ],
                [
                    14.1,
                    13.791,
                    13.294,
                    13.066,
                    13.176,
                    12.787
                ],
                [
                    50.814,
                    50.276,
                    49.058,
                    45.727,
                    50.403,
                    49.679
                ],
                [
                    116.954,
                    124.624,
                    109.42,
                    113.395,
                    118.641,
                    120.176
                ],
                [
                    100.022,
                    121.132,
                    109.273,
                    118.068,
                    117.521,
                    117.317
                ]
            ]
        ])
    tsc_values = np.asarray([
            [
                [
                    -18.947,
                    -22.549,
                    -19.871,
                    -19.987,
                    -23.025,
                    -21.052
                ],
                [
                    -15.778,
                    -14.47,
                    -16.21,
                    -14.989,
                    -15.742,
                    -14.47
                ],
                [
                    -12.018,
                    -12.256,
                    -10.432,
                    -11.026,
                    -11.143,
                    -11.158
                ],
                [
                    -12.058,
                    -10.676,
                    -12.691,
                    -12.36,
                    -12.299,
                    -10.515
                ],
                [
                    -15.609,
                    -12.397,
                    -12.565,
                    -15.907,
                    -13.534,
                    -13.301
                ],
                [
                    -16.229,
                    -17.706,
                    -17.563,
                    -17.606,
                    -18.141,
                    -16.148
                ]
            ],
            [
                [
                    -19.751,
                    -19.02,
                    -20.419,
                    -21.82,
                    -17.41,
                    -18.423
                ],
                [
                    -28.229,
                    -25.375,
                    -25.628,
                    -23.835,
                    -27.182,
                    -24.023
                ],
                [
                    -23.314,
                    -27.709,
                    -25.349,
                    -22.924,
                    -22.674,
                    -24.966
                ],
                [
                    -11.333,
                    -11.886,
                    -11.374,
                    -12.173,
                    -10.242,
                    -14.129
                ],
                [
                    -6.331,
                    -7.695,
                    -7.136,
                    -7.679,
                    -6.596,
                    -6.386
                ],
                [
                    -5.884,
                    -5.297,
                    -8.417,
                    -6.065,
                    -5.548,
                    -6.892
                ]
            ],
            [
                [
                    -15.27,
                    -15.552,
                    -16.876,
                    -12.551,
                    -17.799,
                    -15.74
                ],
                [
                    -27.15,
                    -29.132,
                    -28.903,
                    -27.759,
                    -26.549,
                    -28.25
                ],
                [
                    -20.482,
                    -24.803,
                    -25.203,
                    -26.651,
                    -25.716,
                    -23.832
                ],
                [
                    -4.889,
                    -2.255,
                    -9.789,
                    -10.069,
                    -9.529,
                    -13.052
                ],
                [
                    -4.84,
                    -3.765,
                    -3.918,
                    -2.206,
                    -4.146,
                    -3.931
                ],
                [
                    -2.555,
                    -3.236,
                    -2.61,
                    -3.818,
                    -4.013,
                    -3.937
                ]
            ],
            [
                [
                    -17.731,
                    -16.517,
                    -16.181,
                    -15.016,
                    -18.691,
                    -17.558
                ],
                [
                    -17.361,
                    -17.43,
                    -16.456,
                    -16.057,
                    -16.52,
                    -16.893
                ],
                [
                    -13.362,
                    -5.306,
                    -9.115,
                    -12.094,
                    -3.433,
                    -10.928
                ],
                [
                    -8.691,
                    -9.353,
                    -9.352,
                    -10.194,
                    -12.814,
                    -11.486
                ],
                [
                    -11.16,
                    -4.984,
                    -4.365,
                    -4.636,
                    -8.842,
                    -7.316
                ],
                [
                    -11.958,
                    -5.867,
                    -9.208,
                    -6.672,
                    -6.437,
                    -8.976
                ]
            ],
            [
                [
                    -19.119,
                    -21.013,
                    -22.027,
                    -18.489,
                    -16.898,
                    -19.971
                ],
                [
                    -9.71,
                    -7.965,
                    -7.474,
                    -9.096,
                    -9.964,
                    -6.601
                ],
                [
                    -6.958,
                    -2.616,
                    -3.682,
                    -4.656,
                    -5.021,
                    -3.976
                ],
                [
                    -7.188,
                    -6.314,
                    -7.361,
                    -7.186,
                    -5.851,
                    -2.389
                ],
                [
                    -19.097,
                    -18.032,
                    -9.169,
                    -6.435,
                    -20.175,
                    -12.576
                ],
                [
                    -30.999,
                    -24.074,
                    -24.572,
                    -23.45,
                    -29.649,
                    -22.429
                ]
            ],
            [
                [
                    -17.377,
                    -17.559,
                    -19.157,
                    -15.264,
                    -16.336,
                    -18.369
                ],
                [
                    -4.394,
                    -6.168,
                    -4.519,
                    -5.264,
                    -6.076,
                    -3.754
                ],
                [
                    -2.102,
                    -3.785,
                    -3.28,
                    -2.439,
                    -2.424,
                    -3.417
                ],
                [
                    -7.389,
                    -3.41,
                    -5.002,
                    -9.065,
                    -6.738,
                    -6.016
                ],
                [
                    -17.965,
                    -25.787,
                    -24.513,
                    -20.441,
                    -17.686,
                    -22.043
                ],
                [
                    -28.803,
                    -31.01,
                    -32.605,
                    -28.362,
                    -36.305,
                    -36.09
                ]
            ]
        ])
    masc_values = np.asarray([
            [
                [
                    9.378,
                    9.259,
                    8.682,
                    9.52,
                    9.08,
                    9.147
                ],
                [
                    6.963,
                    6.246,
                    6.625,
                    6.414,
                    6.212,
                    6.235
                ],
                [
                    5.232,
                    5.477,
                    4.701,
                    4.948,
                    5.113,
                    4.905
                ],
                [
                    5.704,
                    5.831,
                    5.843,
                    5.823,
                    5.853,
                    5.843
                ],
                [
                    6.939,
                    6.836,
                    7.097,
                    7.186,
                    6.608,
                    6.966
                ],
                [
                    7.624,
                    8.225,
                    7.968,
                    8.164,
                    7.593,
                    8.029
                ]
            ],
            [
                [
                    8.178,
                    8.168,
                    8.356,
                    8.485,
                    8.026,
                    8.13
                ],
                [
                    11.706,
                    11.891,
                    11.933,
                    11.481,
                    12.38,
                    11.79
                ],
                [
                    11.502,
                    11.113,
                    11.068,
                    11.236,
                    10.994,
                    11.519
                ],
                [
                    7.771,
                    8.054,
                    8.301,
                    7.788,
                    7.858,
                    7.722
                ],
                [
                    4.671,
                    4.512,
                    4.888,
                    4.613,
                    4.512,
                    4.862
                ],
                [
                    3.078,
                    3.114,
                    3.306,
                    3.102,
                    3.169,
                    3.338
                ]
            ],
            [
                [
                    6.722,
                    6.895,
                    7.231,
                    6.252,
                    7.029,
                    6.682
                ],
                [
                    13.003,
                    13.203,
                    13.163,
                    13.437,
                    13.096,
                    13.792
                ],
                [
                    16.17,
                    16.744,
                    16.804,
                    16.22,
                    16.521,
                    16.51
                ],
                [
                    11.005,
                    10.953,
                    11.145,
                    11.26,
                    11.205,
                    10.792
                ],
                [
                    3.883,
                    3.847,
                    4.018,
                    3.983,
                    3.802,
                    3.965
                ],
                [
                    3.18,
                    3.26,
                    3.277,
                    3.476,
                    3.362,
                    3.269
                ]
            ],
            [
                [
                    7.205,
                    7.245,
                    7.318,
                    6.988,
                    7.45,
                    7.153
                ],
                [
                    9.311,
                    10.09,
                    10.013,
                    10.568,
                    9.824,
                    10.26
                ],
                [
                    13.021,
                    12.614,
                    13.869,
                    12.943,
                    11.687,
                    12.466
                ],
                [
                    9.616,
                    10.693,
                    9.562,
                    10.295,
                    10.852,
                    9.874
                ],
                [
                    8.204,
                    7.174,
                    7.128,
                    7.753,
                    7.835,
                    8.793
                ],
                [
                    10.676,
                    10.481,
                    11.009,
                    10.458,
                    11.728,
                    10.662
                ]
            ],
            [
                [
                    8.171,
                    8.201,
                    8.237,
                    8.57,
                    7.806,
                    8.331
                ],
                [
                    5.449,
                    5.746,
                    5.761,
                    6.044,
                    5.852,
                    5.73
                ],
                [
                    4.76,
                    5.083,
                    4.273,
                    4.774,
                    4.796,
                    4.812
                ],
                [
                    8.921,
                    8.709,
                    8.005,
                    8.013,
                    8.079,
                    8.683
                ],
                [
                    21.118,
                    21.716,
                    20.621,
                    20.331,
                    22.74,
                    20.872
                ],
                [
                    24.536,
                    26.055,
                    26.128,
                    25.322,
                    26.739,
                    26.517
                ]
            ],
            [
                [
                    8.024,
                    7.802,
                    7.6,
                    7.035,
                    7.407,
                    7.392
                ],
                [
                    2.84,
                    3.084,
                    2.841,
                    3.025,
                    2.91,
                    2.855
                ],
                [
                    3.366,
                    3.292,
                    3.174,
                    3.119,
                    3.145,
                    3.053
                ],
                [
                    12.131,
                    12.003,
                    11.712,
                    10.917,
                    12.033,
                    11.86
                ],
                [
                    27.921,
                    29.752,
                    26.122,
                    27.071,
                    28.323,
                    28.69
                ],
                [
                    23.878,
                    28.918,
                    26.087,
                    28.187,
                    28.056,
                    28.007
                ]
            ]
        ])
    msc_values = np.asarray([
            [
                [
                    -4.523,
                    -5.383,
                    -4.744,
                    -4.772,
                    -5.497,
                    -5.026
                ],
                [
                    -3.767,
                    -3.455,
                    -3.87,
                    -3.578,
                    -3.758,
                    -3.455
                ],
                [
                    -2.869,
                    -2.926,
                    -2.49,
                    -2.632,
                    -2.66,
                    -2.664
                ],
                [
                    -2.879,
                    -2.549,
                    -3.03,
                    -2.951,
                    -2.936,
                    -2.51
                ],
                [
                    -3.726,
                    -2.959,
                    -3.0,
                    -3.797,
                    -3.231,
                    -3.175
                ],
                [
                    -3.874,
                    -4.227,
                    -4.193,
                    -4.203,
                    -4.331,
                    -3.855
                ]
            ],
            [
                [
                    -4.715,
                    -4.541,
                    -4.875,
                    -5.209,
                    -4.156,
                    -4.398
                ],
                [
                    -6.739,
                    -6.058,
                    -6.118,
                    -5.69,
                    -6.489,
                    -5.735
                ],
                [
                    -5.566,
                    -6.615,
                    -6.052,
                    -5.473,
                    -5.413,
                    -5.96
                ],
                [
                    -2.706,
                    -2.838,
                    -2.715,
                    -2.906,
                    -2.445,
                    -3.373
                ],
                [
                    -1.511,
                    -1.837,
                    -1.704,
                    -1.833,
                    -1.575,
                    -1.525
                ],
                [
                    -1.405,
                    -1.265,
                    -2.009,
                    -1.448,
                    -1.324,
                    -1.645
                ]
            ],
            [
                [
                    -3.646,
                    -3.713,
                    -4.029,
                    -2.996,
                    -4.249,
                    -3.758
                ],
                [
                    -6.482,
                    -6.955,
                    -6.9,
                    -6.627,
                    -6.338,
                    -6.744
                ],
                [
                    -4.89,
                    -5.921,
                    -6.017,
                    -6.362,
                    -6.139,
                    -5.69
                ],
                [
                    -1.167,
                    -0.538,
                    -2.337,
                    -2.404,
                    -2.275,
                    -3.116
                ],
                [
                    -1.155,
                    -0.899,
                    -0.935,
                    -0.527,
                    -0.99,
                    -0.938
                ],
                [
                    -0.61,
                    -0.773,
                    -0.623,
                    -0.912,
                    -0.958,
                    -0.94
                ]
            ],
            [
                [
                    -4.233,
                    -3.943,
                    -3.863,
                    -3.585,
                    -4.462,
                    -4.192
                ],
                [
                    -4.145,
                    -4.161,
                    -3.929,
                    -3.833,
                    -3.944,
                    -4.033
                ],
                [
                    -3.19,
                    -1.267,
                    -2.176,
                    -2.887,
                    -0.819,
                    -2.609
                ],
                [
                    -2.075,
                    -2.233,
                    -2.233,
                    -2.434,
                    -3.059,
                    -2.742
                ],
                [
                    -2.664,
                    -1.19,
                    -1.042,
                    -1.107,
                    -2.111,
                    -1.747
                ],
                [
                    -2.855,
                    -1.401,
                    -2.198,
                    -1.593,
                    -1.537,
                    -2.143
                ]
            ],
            [
                [
                    -4.564,
                    -5.016,
                    -5.259,
                    -4.414,
                    -4.034,
                    -4.768
                ],
                [
                    -2.318,
                    -1.902,
                    -1.784,
                    -2.172,
                    -2.379,
                    -1.576
                ],
                [
                    -1.661,
                    -0.625,
                    -0.879,
                    -1.112,
                    -1.199,
                    -0.949
                ],
                [
                    -1.716,
                    -1.507,
                    -1.757,
                    -1.716,
                    -1.397,
                    -0.57
                ],
                [
                    -4.559,
                    -4.305,
                    -2.189,
                    -1.536,
                    -4.816,
                    -3.002
                ],
                [
                    -7.4,
                    -5.747,
                    -5.866,
                    -5.598,
                    -7.078,
                    -5.355
                ]
            ],
            [
                [
                    -4.149,
                    -4.192,
                    -4.573,
                    -3.644,
                    -3.9,
                    -4.385
                ],
                [
                    -1.049,
                    -1.472,
                    -1.079,
                    -1.257,
                    -1.451,
                    -0.896
                ],
                [
                    -0.502,
                    -0.904,
                    -0.783,
                    -0.582,
                    -0.579,
                    -0.816
                ],
                [
                    -1.764,
                    -0.814,
                    -1.194,
                    -2.164,
                    -1.609,
                    -1.436
                ],
                [
                    -4.289,
                    -6.156,
                    -5.852,
                    -4.88,
                    -4.222,
                    -5.262
                ],
                [
                    -6.876,
                    -7.403,
                    -7.784,
                    -6.771,
                    -8.667,
                    -8.616
                ]
            ]
        ])
    points = np.asarray([
        [
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0
        ],
        [
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0
        ],
        [
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0
        ]
    ])

        
    all_values = [tasc_values, tsc_values, masc_values, msc_values]
    red_points = np.asarray([
        [
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0
        ],
        [
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0
        ]])
    for j in range(len(labels)):
        label = labels[j]
        label_values = all_values[j]
        directory = f"plots/preliminary_tests/cost_function/{label}"
        os.makedirs(directory, exist_ok=True)
        for i in range(len(points[0])):
            values = label_values[i,:,:]
            title = f"{label} for 3D cost function (S-Rank = 1, num_data_points = 1) \n where $x_1 = {points[0][i]}$"
            file_name = f"cost_function/{label}/QNN_cost_{label}_x1={points[0][i]}_surface"
            file_name2 = f"cost_function/{label}/QNN_cost_{label}_x1={points[0][i]}_circle"
            plot_2D_surface(red_points, values, label, title, file_name)
            plot_2D_overlapping_circles(red_points, values, label, title, file_name2)
        print("Outliers for ", label)
        a, b = determine_outliers_in_grid(points, label_values)
        print("Points", a)
        print("Values", b)

def print_array_summary(array, file=""):
    if file!="":
        f = open(file, "a")
        f.write("summary:\n")
        f.write("median "+str(np.median(array))+"\n")
        f.write("mean "+str(np.mean(array))+"\n")
        f.write("std "+str(np.std(array))+"\n")
        f.write("variance "+str(np.var(array))+"\n")
        f.write("max "+str(np.max(array))+"\n")
        f.write("min "+str(np.min(array))+"\n")
        f.close()
    else:
        print("summary:")
        print("median", np.median(array))
        print("mean", np.mean(array))
        print("std", np.std(array))
        print("variance", np.var(array))
        print("max", np.max(array))
        print("min", np.min(array))

def one_iteration_grid_TASC(function, lower_left, stepsize, directory="", iteration=0, N=3,plots=False):
    '''
        Calculates TASC of a function for every point on a grid and outputs corresponding information, such as outliers and plots.
    '''
    #sys.stdout = open("plots/preliminary_tests/cost_function/third_test_2025-03-11/result.txt", 'w')
    label = "TASC"
    start = time.time()
    print("Generating grid point array...")
    points = generate_grid_point_array(stepsize,lower_left,N)
    print("Calculating TASC landscape...")
    tasc_landscape,tsc_landscape,masc_landscape,msc_landscape = calc_landscape_tasc(function, points, r=stepsize)
    print("Number of Grid Points", N)
    #print("Grid Points", points.tolist())
    #print("TASC", tasc_landscape.tolist())
    print("TASC")
    print_array_summary(tasc_landscape)
    outlier_points, outlier_values = determine_outliers_in_grid(points, tasc_landscape)
    print("Outlier Points", outlier_points)
    print("Outlier TASC Values", outlier_values)
    
    print("TSC")
    print_array_summary(tsc_landscape)
    outlier_points, outlier_values = determine_outliers_in_grid(points, tsc_landscape)
    print("Outlier Points", outlier_points)
    print("Outlier TSC Values", outlier_values)

    print("MASC")
    print_array_summary(masc_landscape)
    outlier_points, outlier_values = determine_outliers_in_grid(points, masc_landscape)
    print("Outlier Points", outlier_points)
    print("Outlier MASC Values", outlier_values)

    print("MSC")
    print_array_summary(msc_landscape)
    outlier_points, outlier_values = determine_outliers_in_grid(points, msc_landscape)
    print("Outlier Points", outlier_points)
    print("Outlier MSC Values", outlier_values)

    dim = len(lower_left)
    elapsed_time = time.time()-start
    print("Time (minutes)", np.round(elapsed_time/60,3))
    #sys.stdout.close()
    if dim == 3 and plots==True:
        file_name = f"QNN_3D_cost_TASC_iteration_{iteration}_"
        red_points = np.asarray([points[1,:], points[2,:]])
        for i in range(len(points[0])):
                dir = f"plots/preliminary_tests/{directory}/Iteration{iteration}"
                os.makedirs(dir, exist_ok=True)
                values = tasc_landscape[i,:,:]
                title = f"{label} for 3D cost function (S-Rank = 2, num_data_points = 1) \n where $x_1 = {points[0][i]}$"
                file_name = f"{directory}/Iteration{iteration}/QNN_3D_cost_{label}_iteration{iteration}_x1={points[0][i]}_surface"
                file_name2 = f"{directory}/Iteration{iteration}/QNN_3D_cost_{label}_iteration{iteration}_x1={points[0][i]}_circle"
                plot_2D_surface(red_points, values, label, title, file_name)
                plot_2D_overlapping_circles(red_points, values, label, title, file_name2)
    return elapsed_time

def test_3D_CostFunc():
    num_qubits = 1
    s_ranks = [1,2,3,4]
    num_data_points = [1,2,3,4]
    data_types = [1,2,3,4]
    c = np.asarray([0,0,0,0,0,0])
    times = []
    for s in s_ranks:
        for ndp in num_data_points:
            for type in data_types:
                print(f"Schmidt rank = {s}, number of data points = {ndp}, data type = {type}")
                try:
                    cost_function = get_basic_6D_cost_function(s, ndp, type)
                    time = one_iteration_grid_TASC(cost_function, c, 1, N=4)
                    times.append(time)
                except:
                    print("An Exception occured")
    print("average time", np.round(np.mean(times)/60,3))



if __name__ == "__main__":
    #cost_func_grid_TASC()
    #plot_all_tasc_grids_from_run0()
    directory = "plots/preliminary_tests/cost_function/second_test_2025-03-10"
    os.makedirs(directory, exist_ok=True)
    directory = "cost_function/third_test_2025-03-10/"
    cost_function = get_basic_6D_cost_function(s_rank=4, ndp=1)
    c = np.asarray([0,0,0,0,0,0]) 
    N = 5
    stepsize = 1.75
    #print("QNN 6D, Schmidt Rank = 4, Num Data Points = 1")
    #one_iteration_grid_TASC(cost_function, c, stepsize, directory, 0, N=N)
    test_3D_CostFunc()
"""     points = np.asarray([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    tasc_landscape = np.asarray([[[43.894, 40.24, 38.468, 39.819, 38.892, 43.963, 40.213], [43.486, 42.007, 41.373, 40.635, 39.099, 39.701, 42.173], [46.381, 46.203, 45.982, 45.675, 46.466, 46.319, 47.529], [49.151, 55.274, 50.106, 49.263, 51.866, 53.366, 55.004], [49.712, 47.465, 48.37, 49.196, 50.233, 49.207, 48.479], [41.417, 43.551, 42.102, 44.566, 43.849, 40.312, 41.027], [43.361, 41.674, 40.922, 42.545, 41.902, 41.052, 39.519]], [[21.613, 22.246, 21.684, 23.082, 21.903, 21.584, 21.144], [35.766, 33.824, 34.549, 34.761, 35.67, 34.35, 34.066], [43.133, 45.103, 44.306, 41.932, 43.607, 42.497, 43.188], [44.868, 44.266, 44.502, 44.378, 44.257, 46.278, 46.125], [38.249, 32.798, 32.251, 33.089, 38.97, 34.298, 34.206], [22.468, 22.328, 21.779, 21.895, 22.498, 22.184, 21.564], [23.878, 26.859, 27.478, 26.353, 25.384, 26.299, 24.282]], [[9.046, 8.726, 8.057, 8.614, 8.031, 8.535, 8.394], [16.451, 15.805, 15.98, 15.986, 15.036, 15.651, 15.279], [27.711, 27.78, 29.232, 26.757, 29.107, 28.733, 28.519], [33.383, 31.947, 31.139, 31.113, 33.032, 31.345, 35.982], [20.826, 19.039, 19.099, 18.738, 19.512, 19.299, 17.955], [10.762, 9.793, 9.091, 10.114, 10.127, 9.327, 9.884], [14.281, 14.029, 14.395, 13.591, 14.377, 13.954, 14.269]], [[7.536, 7.079, 7.228, 7.037, 7.173, 7.178, 7.056], [7.577, 7.485, 7.275, 7.942, 7.754, 8.089, 6.944], [13.367, 13.888, 13.325, 13.803, 13.944, 14.068, 13.099], [20.436, 20.188, 20.017, 21.397, 20.165, 23.2, 22.135], [20.161, 23.754, 22.377, 21.538, 23.253, 23.356, 21.957], [23.588, 24.75, 24.467, 24.967, 23.029, 24.071, 23.633], [18.351, 18.577, 17.24, 17.151, 17.378, 18.374, 18.037]], [[13.046, 13.319, 12.81, 13.472, 14.924, 12.769, 13.63], [8.595, 8.58, 8.86, 9.184, 9.023, 8.873, 9.064], [16.017, 16.016, 15.872, 17.042, 16.347, 16.05, 16.96], [45.317, 42.991, 41.944, 39.521, 43.433, 43.537, 43.985], [70.239, 72.783, 66.654, 64.94, 70.48, 68.006, 67.675], [66.576, 60.05, 60.11, 66.292, 64.318, 64.018, 66.184], [33.934, 33.498, 34.671, 33.153, 30.695, 32.215, 33.064]], [[30.771, 28.343, 31.672, 31.967, 27.238, 30.725, 30.681], [22.946, 21.633, 22.442, 21.626, 21.784, 23.149, 21.265], [33.388, 32.145, 33.344, 31.084, 33.842, 32.998, 32.196], [74.317, 76.495, 75.007, 77.456, 75.965, 73.286, 72.853], [101.51, 99.791, 100.78, 106.688, 100.815, 98.739, 106.145], [82.637, 90.912, 78.336, 81.93, 81.388, 80.929, 86.452], [46.239, 46.792, 48.535, 44.473, 47.214, 44.543, 45.858]], [[44.697, 43.86, 44.558, 41.512, 44.194, 41.511, 45.293], [38.64, 43.091, 45.266, 39.058, 40.865, 40.781, 41.789], [50.867, 54.11, 52.918, 53.929, 52.463, 51.453, 53.971], [70.992, 71.46, 68.741, 69.988, 73.128, 72.621, 74.914], [66.958, 70.332, 72.951, 70.078, 73.868, 72.905, 71.46], [55.554, 54.561, 52.553, 56.959, 59.727, 56.855, 56.635], [49.827, 45.978, 51.264, 45.611, 47.227, 48.616, 48.576]]])
    label = "TASC"
    red_points = np.asarray([points[1,:], points[2,:]])
    for i in range(len(points[0])):
        dir = f"plots/preliminary_tests/cost_function/second_test_2025-03-10/"
        os.makedirs(dir, exist_ok=True)
        values = tasc_landscape[i,:,:]
        title = f"{label} for 3D cost function (S-Rank = 2, num_data_points = 1) \n where $x_1 = {points[0][i]}$"
        file_name = f"{directory}/QNN_3D_cost_{label}_x1={points[0][i]}_surface"
        file_name2 = f"{directory}/QNN_3D_cost_{label}_x1={points[0][i]}_circle"
        plot_2D_surface(red_points, values, label, title, file_name)
        plot_2D_overlapping_circles(red_points, values, label, title, file_name2) """



"""     points = np.asarray([[5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0], [4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0], [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]])
    array = np.asarray([[[0.108, 0.118, 0.115, 0.12, 0.106, 0.111, 0.115, 0.111, 0.115], [0.101, 0.101, 0.101, 0.107, 0.106, 0.104, 0.108, 0.104, 0.102], [0.102, 0.104, 0.102, 0.107, 0.106, 0.105, 0.109, 0.107, 0.107], [0.114, 0.114, 0.11, 0.108, 0.112, 0.111, 0.114, 0.118, 0.112], [0.119, 0.117, 0.114, 0.124, 0.112, 0.124, 0.12, 0.117, 0.124], [0.125, 0.124, 0.124, 0.136, 0.13, 0.127, 0.134, 0.127, 0.12], [0.148, 0.148, 0.15, 0.15, 0.147, 0.15, 0.15, 0.151, 0.149], [0.175, 0.174, 0.188, 0.193, 0.184, 0.18, 0.176, 0.178, 0.179], [0.215, 0.215, 0.23, 0.214, 0.213, 0.209, 0.22, 0.2, 0.218]], [[0.132, 0.139, 0.133, 0.135, 0.145, 0.13, 0.141, 0.142, 0.131], [0.142, 0.148, 0.138, 0.128, 0.136, 0.142, 0.139, 0.137, 0.143], [0.149, 0.153, 0.149, 0.149, 0.146, 0.139, 0.155, 0.149, 0.156], [0.166, 0.154, 0.166, 0.162, 0.162, 0.176, 0.167, 0.169, 0.163], [0.19, 0.191, 0.177, 0.192, 0.181, 0.191, 0.187, 0.172, 0.178], [0.194, 0.209, 0.202, 0.194, 0.2, 0.206, 0.215, 0.206, 0.192], [0.226, 0.228, 0.221, 0.236, 0.221, 0.226, 0.233, 0.228, 0.226], [0.262, 0.271, 0.272, 0.261, 0.262, 0.263, 0.27, 0.267, 0.279], [0.322, 0.318, 0.308, 0.304, 0.297, 0.299, 0.302, 0.276, 0.284]], [[0.198, 0.208, 0.201, 0.212, 0.194, 0.214, 0.213, 0.199, 0.195], [0.204, 0.204, 0.209, 0.211, 0.216, 0.211, 0.226, 0.202, 0.21], [0.225, 0.226, 0.227, 0.223, 0.226, 0.219, 0.238, 0.23, 0.228], [0.244, 0.249, 0.246, 0.254, 0.256, 0.237, 0.243, 0.249, 0.25], [0.25, 0.263, 0.264, 0.267, 0.268, 0.265, 0.261, 0.259, 0.258], [0.271, 0.279, 0.276, 0.268, 0.284, 0.302, 0.263, 0.278, 0.281], [0.304, 0.308, 0.294, 0.307, 0.316, 0.319, 0.31, 0.319, 0.307], [0.349, 0.354, 0.355, 0.365, 0.33, 0.364, 0.342, 0.357, 0.362], [0.386, 0.388, 0.416, 0.38, 0.388, 0.395, 0.393, 0.388, 0.395]], [[0.257, 0.268, 0.262, 0.248, 0.254, 0.263, 0.259, 0.27, 0.257], [0.265, 0.286, 0.265, 0.282, 0.272, 0.261, 0.263, 0.282, 0.276], [0.293, 0.293, 0.292, 0.307, 0.313, 0.295, 0.283, 0.3, 0.293], [0.319, 0.322, 0.332, 0.307, 0.324, 0.312, 0.322, 0.32, 0.322], [0.344, 0.359, 0.354, 0.359, 0.341, 0.356, 0.339, 0.373, 0.333], [0.391, 0.355, 0.374, 0.362, 0.357, 0.362, 0.344, 0.37, 0.367], [0.359, 0.386, 0.38, 0.379, 0.395, 0.377, 0.373, 0.369, 0.388], [0.413, 0.416, 0.441, 0.42, 0.424, 0.419, 0.409, 0.428, 0.414], [0.5, 0.449, 0.517, 0.481, 0.447, 0.493, 0.491, 0.489, 0.456]], [[0.358, 0.335, 0.353, 0.384, 0.352, 0.368, 0.36, 0.368, 0.349], [0.344, 0.363, 0.342, 0.365, 0.341, 0.348, 0.354, 0.346, 0.34], [0.384, 0.372, 0.379, 0.388, 0.376, 0.389, 0.357, 0.362, 0.378], [0.403, 0.385, 0.402, 0.428, 0.418, 0.418, 0.418, 0.439, 0.403], [0.437, 0.423, 0.41, 0.425, 0.432, 0.45, 0.431, 0.421, 0.428], [0.411, 0.394, 0.439, 0.4, 0.43, 0.427, 0.406, 0.417, 0.426], [0.42, 0.424, 0.385, 0.427, 0.441, 0.443, 0.423, 0.42, 0.416], [0.506, 0.457, 0.462, 0.5, 0.501, 0.504, 0.475, 0.497, 0.488], [0.539, 0.544, 0.581, 0.585, 0.57, 0.58, 0.549, 0.569, 0.546]], [[0.541, 0.481, 0.496, 0.494, 0.454, 0.509, 0.499, 0.525, 0.479], [0.477, 0.482, 0.46, 0.463, 0.446, 0.464, 0.474, 0.453, 0.474], [0.476, 0.512, 0.486, 0.476, 0.489, 0.509, 0.479, 0.503, 0.486], [0.516, 0.522, 0.522, 0.522, 0.533, 0.49, 0.501, 0.497, 0.507], [0.48, 0.511, 0.51, 0.519, 0.513, 0.528, 0.506, 0.5, 0.505], [0.471, 0.47, 0.48, 0.487, 0.503, 0.514, 0.49, 0.517, 0.516], [0.483, 0.483, 0.449, 0.508, 0.457, 0.496, 0.502, 0.482, 0.464], [0.522, 0.534, 0.55, 0.523, 0.542, 0.556, 0.525, 0.577, 0.525], [0.548, 0.575, 0.604, 0.561, 0.615, 0.635, 0.596, 0.607, 0.597]], [[0.615, 0.627, 0.626, 0.617, 0.603, 0.625, 0.624, 0.65, 0.627], [0.586, 0.636, 0.602, 0.587, 0.619, 0.596, 0.62, 0.653, 0.59], [0.582, 0.585, 0.587, 0.632, 0.605, 0.61, 0.565, 0.584, 0.638], [0.601, 0.607, 0.612, 0.578, 0.608, 0.6, 0.649, 0.622, 0.625], [0.627, 0.58, 0.611, 0.647, 0.608, 0.57, 0.627, 0.618, 0.6], [0.593, 0.558, 0.576, 0.591, 0.573, 0.585, 0.571, 0.577, 0.578], [0.567, 0.585, 0.564, 0.579, 0.592, 0.586, 0.563, 0.598, 0.566], [0.62, 0.651, 0.634, 0.638, 0.617, 0.642, 0.603, 0.633, 0.621], [0.692, 0.716, 0.665, 0.664, 0.664, 0.655, 0.659, 0.619, 0.678]], [[0.804, 0.793, 0.771, 0.765, 0.789, 0.795, 0.812, 0.771, 0.803], [0.74, 0.774, 0.793, 0.706, 0.736, 0.742, 0.734, 0.756, 0.743], [0.734, 0.699, 0.724, 0.698, 0.718, 0.711, 0.712, 0.703, 0.728], [0.688, 0.753, 0.74, 0.73, 0.703, 0.722, 0.679, 0.735, 0.747], [0.704, 0.651, 0.667, 0.652, 0.692, 0.716, 0.691, 0.643, 0.7], [0.628, 0.657, 0.617, 0.646, 0.638, 0.624, 0.607, 0.664, 0.622], [0.637, 0.606, 0.606, 0.618, 0.658, 0.63, 0.629, 0.648, 0.637], [0.662, 0.681, 0.642, 0.653, 0.634, 0.649, 0.651, 0.59, 0.66], [0.707, 0.67, 0.686, 0.664, 0.69, 0.697, 0.642, 0.695, 0.702]], [[0.903, 0.896, 0.934, 0.937, 0.932, 0.896, 0.872, 0.908, 0.901], [0.821, 0.84, 0.854, 0.801, 0.838, 0.806, 0.814, 0.866, 0.838], [0.776, 0.836, 0.8, 0.768, 0.797, 0.855, 0.815, 0.806, 0.78], [0.808, 0.782, 0.83, 0.798, 0.755, 0.761, 0.756, 0.798, 0.781], [0.795, 0.744, 0.73, 0.782, 0.71, 0.745, 0.758, 0.768, 0.741], [0.725, 0.722, 0.71, 0.709, 0.697, 0.681, 0.709, 0.733, 0.706], [0.679, 0.678, 0.643, 0.644, 0.661, 0.677, 0.688, 0.641, 0.63], [0.656, 0.692, 0.63, 0.663, 0.648, 0.699, 0.643, 0.652, 0.685], [0.672, 0.716, 0.695, 0.7, 0.65, 0.639, 0.693, 0.68, 0.667]]])
    for t in [2.75,2.5,2.25,2]:
        outlier_points, outlier_values = determine_outliers_in_grid(points, array, z_score_threshold=t)
        print("Threshold", t)
        print("Outlier Points", outlier_points)
        print("Outlier TASC Values", outlier_values) """
