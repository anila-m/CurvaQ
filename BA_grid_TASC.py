from datetime import datetime
import json
import os
import time
import numpy as np
from scipy import stats

from metrics import *
from BA_testing_functions import rosen_projection_to_2d
from BA_thesis_plots import plot_2D_overlapping_circles, plot_2D_scatter, plot_2D_surface


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

def calc_landscape_tasc(function, grid_point_array, r=0,no_samples=1000):
    '''
        Calculate total absolute scalar curvature (and other metrics, TSC; MASC; MSC) of the given function 
        in a ball of radius r at every point given in the grid_point_array.

        Args:
            function (callable): function
            grid_point_array (array): array of arrays of node values for grid for every dimension
            r (float): radius of ball, default: r=0 (r will be calculcated to equal stepsize of grid)
            N (int): number of samples in each region, default: 1000
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
        # calculate tasc, tsc, mean absolute sc and mean sc values at this point
        tasc_landscape[idx],tsc_landscape[idx],mean_asc_landscape[idx], mean_sc_landscape[idx],sc_summary, grad_summary, hess_summary = calc_several_scalar_curvature_values(function, r, point,N=no_samples)
        j += 1
    return tasc_landscape, tsc_landscape, mean_asc_landscape, mean_sc_landscape

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
            points (array): nxN array, that defines the points on the grid in each dimension (n=dimensions, N=no of points in each dimension)
            values (array): N^n array, values corresponding to each point on the grid
            z_score_threshold (float): threshold for Z-score outlier detection, default: 3
        
        Return:
            outlier_points (array): points where outlier occurs
            outlier_values (array): list of outliers associated with outlier_points
            outlier_indices (array): list of indices of outliers
            number of outliers with negative Z-Score (int)
            number of outliers with positive Z-Score (int)
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
    # determine whether outlier has positive or negative z-Score
    negative_z_score = Z[Z<-z_score_threshold]
    positive_z_score = Z[Z>z_score_threshold]
    return outlier_points, outlier_values, outlier_indices, len(negative_z_score), len(positive_z_score)

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
        a, b, _,_,_ = determine_outliers_in_grid(points, values)
        print("Points", a)
        print("Values", b)

def plot_all_QNN_tasc_grids():
    '''
        Plots all TASC, TSC, Mean ASC, Mean SC values from 
        results/preliminary_tests/entire_grid/QNN_cost_3D_grid_N=6_2025-03-09.json
        for 3D Cost function.
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
            values = label_values[:,:,i]
            title = f"{label} for 3D cost function (S-Rank = 1, num_data_points = 1) \n where $x_3 = {points[0][i]}$"
            file_name = f"cost_function/{label}/QNN_cost_{label}_x3={points[0][i]}_surface"
            file_name2 = f"cost_function/{label}/QNN_cost_{label}_x3={points[0][i]}_circle"
            plot_2D_surface(red_points, values, label, title, file_name)
            plot_2D_overlapping_circles(red_points, values, label, title, file_name2)
        print("Outliers for ", label)
        a, b, _,_,_ = determine_outliers_in_grid(points, label_values)
        print("Points", a)
        print("Values", b)

def get_array_summary(array, printSummary=False):
    '''
        Summarize data in array (median, mean, standard deviation, variance, maximum and minimum)

        Args:
            array (array): array to be summarized
            printSummary (Boolean): prints summary on console if true, default: False
        Returns:
            list of (median, mean, standard deviation, variance, maximum and minimum)
    '''
    median = np.median(array)
    mean = np.mean(array)
    std = np.std(array)
    variance = np.var(array)
    max = np.max(array)
    min = np.min(array)
    if printSummary:
        print("summary:")
        print("median", median)
        print("mean", mean)
        print("std", std)
        print("variance", variance)
        print("max", max)
        print("min", min)
    return [median, mean, std, variance, max, min]

def one_iteration_grid_TASC(function, lower_left, stepsize, directory="", iteration=0, N=3,plots=False):
    '''
        Calculates TASC of a function for every point on a grid and outputs corresponding information, such as outliers and plots.
        
        Args:
            function (callable): function, 
            lower_left (list): lower left point in grid, 
            stepsize (float): stepsize of grid, 
            directory (String): directory for saving plots, default: "", 
            iteration (int): iteration of TASC calculation (only relevant if the same function and grid is analyzed over multiple iterations), default: 0, 
            N (int): Number of grid points in each direction, default: 3,
            plots (Boolean): if true, plot TASC landscapes (as slices), only possible if dimension is 3, default: False
        Returns:
            elapsed time in seconds
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
    get_array_summary(tasc_landscape)
    outlier_points, outlier_values, _ ,_,_= determine_outliers_in_grid(points, tasc_landscape)
    print("Outlier Points", outlier_points)
    print("Outlier TASC Values", outlier_values)
    
    print("TSC")
    get_array_summary(tsc_landscape)
    outlier_points, outlier_values, _,_,_ = determine_outliers_in_grid(points, tsc_landscape)
    print("Outlier Points", outlier_points)
    print("Outlier TSC Values", outlier_values)

    print("MASC")
    get_array_summary(masc_landscape)
    outlier_points, outlier_values, _,_,_ = determine_outliers_in_grid(points, masc_landscape)
    print("Outlier Points", outlier_points)
    print("Outlier MASC Values", outlier_values)

    print("MSC")
    get_array_summary(msc_landscape)
    outlier_points, outlier_values, _ ,_,_= determine_outliers_in_grid(points, msc_landscape)
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


if __name__ == "__main__":
    plot_all_QNN_tasc_grids()
    
