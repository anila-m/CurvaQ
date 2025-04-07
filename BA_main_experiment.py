from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
import os
import time
import numpy as np
from BA_curvature_util import get_hypersphere_volume, sample_n_ball_uniform
from BA_testing_functions import CostFunction
from BA_grid_TASC import generate_grid_point_array, determine_outliers_in_grid, get_array_summary
from metrics import calc_fourier_density_and_coefficients, calc_scalar_curvature_for_function
from BA_experiment_resources import unitary, inputs_list

datatype_names = ["random", "orthogonal", "linearly dependent", "variable Schmidt rank"]

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
        func_value = self.function(point)
        #print(f"Calculating for point {j}/{no_of_points}", point.tolist())
        # calculate tasc, tsc, mean absolute sc and mean sc values at this point
        idx, tasc, tsc, masc, msc, asc_summary, sc_summary, grad_summary, hess_summary = calc_several_scalar_curvature_values_parallel1(self.function, self.r, point,idx=idx)
        return idx, func_value, tasc, tsc, masc, msc, asc_summary, sc_summary, grad_summary, hess_summary
        #return idx, tasc, tsc, masc, msc


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
    scalar_curvature_landscape, grad_summary, hess_summary = calc_scalar_curvature_for_function(function, sample_points)
    sc_summary = [float(np.median(scalar_curvature_landscape)), float(np.mean(scalar_curvature_landscape)), float(np.min(scalar_curvature_landscape)), float(np.max(scalar_curvature_landscape))]
    asc_summary = [float(np.median(np.absolute(scalar_curvature_landscape))), float(np.mean(np.absolute(scalar_curvature_landscape))), float(np.min(np.absolute(scalar_curvature_landscape))), float(np.max(np.absolute(scalar_curvature_landscape)))]
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
    return idx, np.round(total_absolute_sc,3), np.round(total_sc,3), np.round(mean_asc,3), np.round(mean_sc,3), asc_summary, sc_summary, grad_summary, hess_summary

def main_cost_function_experiment(num_qubits, directory="results/main_experiment/6D_cost"):
    '''
        Main Experiment for 3D or 6D cost functions (32 configurations). Each Configuration: 5 repititions.

        Args:
            num_qubits (int): 1 or 2, (3D: 1, 6D: 2)
            directory (String): directory to save json files in, optional.
    '''
    print(f"CPU Count: {os.cpu_count()}")
    if num_qubits<1 or num_qubits>2:
        raise Exception
    s_rank_range = [1,2]
    N=5
    if num_qubits==2: # case: 6D cost function --> Schmidt rank 3 and 4 also possible
        s_rank_range = [1,2,3,4]
        N=3
    dim = num_qubits*3
    config_id = 0
    date = datetime.today().strftime('%Y-%m-%d')
    for datatype in range(1,5): # (1=random, 2=orthogonal, 3=linearly dependent in H_x, 4= variable schmidt rank)
        for num_of_data_points in range(1,5):
            for s_rank in s_rank_range:
                for run_id in range(1,6):
                    results_dict = {}
                    # get cost function
                    if num_qubits==1:
                        cost_func = CostFunction(num_qubits=num_qubits,s_rank=s_rank, num_data_points=num_of_data_points, data_type=datatype)
                    else:
                        unitary_np = np.asarray(unitary,dtype=complex)
                        inputs = np.asarray(inputs_list[config_id],dtype=complex)
                        cost_func = CostFunction(num_qubits=num_qubits, unitary=unitary_np, inputs=inputs)
                    lowerleft = np.zeros(num_qubits*3)
                    stepsize = 2*np.pi/(N-1)
                    # prepare results dictionary for json file
                    results_dict = {"date": str(date), "config ID": config_id, "run ID": run_id, "dimension": dim, "schmidt Rank": s_rank, "number of data points": num_of_data_points, "data type": datatype_names[datatype-1]}
                    unitary_string = (
                        np.array2string(cost_func.unitary.numpy(), separator=",")
                        .replace("\n", "")
                        .replace(" ", "")
                    )
                    inputs_string = (
                        np.array2string(cost_func.inputs.numpy(), separator=",")
                        .replace("\n", "")
                        .replace(" ", "")
                    )
                    results_dict["unitary"] = unitary_string
                    results_dict["inputs"] = inputs_string
                    results_dict["number of grid points"] = N
                    results_dict["stepsize/radius"] = stepsize
                    results_dict["lower left corner of grid"] = lowerleft.tolist()
                    # compute all relevant values (aka perform experiment)
                    results = one_iteration_grid_TASC_parallel1(cost_func, lowerleft, stepsize,N=N)
                    results_dict.update(results)
                    # save as json file
                    filename = f"{dim}D_cost_config_{config_id}_run_{run_id}_{date}.json"
                    os.makedirs(directory, exist_ok=True)
                    file = open(f"{directory}/{filename}", mode="w")
                    json.dump(results_dict, file, indent=4)
                    # logging
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f"{current_time}: Config {config_id} run {run_id}/5 DONE")
                config_id += 1


                    

def one_iteration_grid_TASC_parallel1(function, lower_left, stepsize,N):
    '''
        Calculates TASC of a function for every point on a grid and outputs corresponding information, such as outliers and plots.
    '''
    results_dict = {} # info: runid, config id, schmidt rank, ndp, datatype, evtl nicht hier einfügen, sondern davor/übergeordnete Funktion
    start = time.time()
    points = generate_grid_point_array(stepsize,lower_left,N)
    results_dict["points"] = points.tolist() # evtl ganze point landscape angeben, damit direkt klar ist welcher Eintrag in den anderenlandscapes zu welchem Eintrag gehört
    
    # calculating
    tasc_landscape,tsc_landscape,masc_landscape,msc_landscape, fd, fcoeff, asc_landscape, sc_landscape, grad_landscape, hess_landscape = calc_landscape_tasc_parallel1(function, points, r=stepsize)
    
    # TASC
    results_dict["TASC"] = {"meaning": "total absolute scalar curvature", "tasc landscape": tasc_landscape.tolist()}
    summary = get_array_summary(tasc_landscape)
    outlier_points, outlier_values = determine_outliers_in_grid(points, tasc_landscape)
    labels = ["median", "mean", "std", "variance", "max", "min"]
    for i in range(len(labels)):
        results_dict["TASC"][labels[i]] = float(summary[i])
    results_dict["TASC"]["outlier points"] = outlier_points.tolist()
    results_dict["TASC"]["outlier values"] = outlier_values.tolist()
    
    # TSC
    results_dict["TSC"] = {"meaning": "total scalar curvature", "tsc landscape": tsc_landscape.tolist()}
    summary = get_array_summary(tsc_landscape)
    outlier_points, outlier_values = determine_outliers_in_grid(points, tsc_landscape)
    labels = ["median", "mean", "std", "variance", "max", "min"]
    for i in range(len(labels)):
        results_dict["TSC"][labels[i]] = float(summary[i])
    results_dict["TSC"]["outlier points"] = outlier_points.tolist()
    results_dict["TSC"]["outlier values"] = outlier_values.tolist()

    # MASC
    results_dict["MASC"] = {"meaning": "mean absolute scalar curvature", "masc landscape": masc_landscape.tolist()}
    summary = get_array_summary(masc_landscape)
    outlier_points, outlier_values = determine_outliers_in_grid(points, masc_landscape)
    labels = ["median", "mean", "std", "variance", "max", "min"]
    for i in range(len(labels)):
        results_dict["MASC"][labels[i]] = float(summary[i])
    results_dict["MASC"]["outlier points"] = outlier_points.tolist()
    results_dict["MASC"]["outlier values"] = outlier_values.tolist()

    # MSC
    results_dict["MSC"] = {"meaning": "mean scalar curvature", "msc landscape": msc_landscape.tolist()}
    summary = get_array_summary(msc_landscape)
    outlier_points, outlier_values = determine_outliers_in_grid(points, msc_landscape)
    labels = ["median", "mean", "std", "variance", "max", "min"]
    for i in range(len(labels)):
        results_dict["MSC"][labels[i]] = float(summary[i])
    results_dict["MSC"]["outlier points"] = outlier_points.tolist()
    results_dict["MSC"]["outlier values"] = outlier_values.tolist()
    
    # Fourier
    results_dict["fourier density"] = float(fd)
    fcoeff_string = (
                        np.array2string(fcoeff, separator=",")
                        .replace("\n", "")
                        .replace(" ", "")
                    )
    results_dict["fourier coefficients"] = fcoeff_string
    
    # ASC
    results_dict["ASC"] = {"meaning": "Summary of absolute scalar curvature values at each point", "order of values": "[median, mean, min, max]", "asc summary landscape": asc_landscape.tolist()}

    # SC
    results_dict["SC"] = {"meaning": "Summary of scalar curvature values at each point", "order of values": "[median, mean, min, max]", "sc summary landscape": sc_landscape.tolist()}

    # Gradients
    results_dict["Gradient"] = {"meaning": "Summary of euclidean norm of gradients at each point", "order of values at each point": "[median, mean, min, max]", "gradient norm summary landscape": grad_landscape.tolist()}
    
    # Hessians
    results_dict["Hessian"] = {"meaning": "Summary of Frobenius norm of Hessians at each point", "order of values": "[median, mean, min, max]", "hessian norm summary landscape": hess_landscape.tolist()}

    # Time
    elapsed_time = time.time()-start
    results_dict["elapsed time (min)"] = float(np.round(elapsed_time/60,3))

    # garbage collector
    del sc_landscape
    del grad_landscape
    del hess_landscape
    del tasc_landscape
    del tsc_landscape
    del masc_landscape
    del msc_landscape
    return results_dict

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
    landscape_4_shape = landscape_shape.copy()
    landscape_4_shape.append(4)
    landscape_4_shape = tuple(landscape_4_shape)
    landscape_shape = tuple(landscape_shape)
    tasc_landscape = np.empty(landscape_shape)
    tsc_landscape = np.empty(landscape_shape)
    mean_asc_landscape = np.empty(landscape_shape)
    mean_sc_landscape = np.empty(landscape_shape)
    cost_landscape = np.empty(landscape_shape)
    asc_landscape = np.empty(landscape_4_shape)
    sc_landscape = np.empty(landscape_4_shape)
    grad_landscape = np.empty(landscape_4_shape)
    hess_landscape = np.empty(landscape_4_shape)
    #concurrency starts here
    engine = Engine(function, r, grid_point_array)
    cpu_count = os.cpu_count()
    assert cpu_count is not None
    with ProcessPoolExecutor(max_workers=cpu_count) as exe:
        futures = [exe.submit(engine, idx) for idx,_ in np.ndenumerate(tasc_landscape)]            
        for future in as_completed(futures):
            # get the result for the next completed task
            idx, cost, tasc, tsc, masc, msc, asc_summary, sc_summary, grad_summary, hess_summary = future.result()# blocks
            tasc_landscape[idx],tsc_landscape[idx],mean_asc_landscape[idx], mean_sc_landscape[idx] = tasc, tsc, masc, msc
            cost_landscape[idx] = cost
            asc_landscape[idx] = asc_summary
            sc_landscape[idx] = sc_summary
            grad_landscape[idx] = grad_summary
            hess_landscape[idx] = hess_summary
    exe.shutdown(wait=True, cancel_futures=False)
    fd, fcoeff = calc_fourier_density_and_coefficients(cost_landscape)
    return tasc_landscape, tsc_landscape, mean_asc_landscape, mean_sc_landscape, fd, fcoeff, asc_landscape, sc_landscape, grad_landscape, hess_landscape

def single_cost_test():
    num_qubits = 1
    s_rank = 2
    ndp = 2
    data_type = 4
    cost_func = CostFunction(num_qubits=num_qubits,s_rank=s_rank, num_data_points=ndp, data_type=data_type)
    ll = np.zeros(num_qubits*3)
    N=2
    stepsize = 2*np.pi/(N-1)
    print("--------------------------------------------")
    print("NO CONCURRENCY")
    print("--------------------------------------------")
    #one_iteration_grid_TASC(cost_func, lower_left=ll, stepsize=stepsize, N=5)
    print("Time (hours) ...")
    print("--------------------------------------------")
    print("WITH CONCURRENCY")
    print("--------------------------------------------")
    one_iteration_grid_TASC_parallel1(cost_func,lower_left=ll,stepsize=stepsize,N=N, info=f"N={N}, dim={num_qubits*3}, s_rank={s_rank}, num_data_points={ndp}, data_type={data_type}")



if __name__=="__main__":
    directory="results/main_experiment/6D_cost"
    main_cost_function_experiment(num_qubits=2,directory=directory)
