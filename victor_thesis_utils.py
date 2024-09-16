import torch
import numpy as np
from data import *
import numpy as np
from utils import *

def calc_combined_std(list_of_std):
    """calculates combined stdv of multiple stdv values

    Args:
        list_of_std (list): a list of standard deviations

    Returns:
        float: combined standard deviation
    """
    sq_sum = 0
    for std in list_of_std:
        sq_sum += std**2
    return np.sqrt(sq_sum/len(list_of_std))

def get_meta_for_mode(mode, data, min_val, max_val, titles, o, gate_name, ansatz):
    """function which calculates parameters for a 2d pyplot representing the data in whatever mode you want

    Args:
        mode (string): describes what kind of data you want to represent (default -> normal landscape, grad -> gradient magnitudes, log_scale -> logarithmic scale for coloring)

    Returns:
        different parameters for the pyplot
    """
    low_threshold = 0.000000001
    if mode == "default":
        c_map = "plasma"
        sup_title = f"Loss Landscapes for {ansatz}($\\phi,\\lambda)$ Approximating {gate_name} for Different Datasets"
        title = titles[o]
        v_min = min(min_val, 0)
        v_max = max(max_val, 1)
    elif mode == "grad":
        c_map = "winter"
        sup_title = "Gradient Magnitudes"
        # average gradient magnitude adjusted for sample frequency
        title = f"GM Score: {np.round(np.average(data)*len(data), 2)}"
        v_min = min(min_val, 0)
        v_max = math.ceil(max_val * 100.0) / 100.0
    elif mode == "log_scale":
        v_max = 1
        v_min = low_threshold
        c_map = "Greys"
        if min_val < low_threshold:
            min_text = "< 0.000000001"
        else:
            min_text = f"= {np.round(min_val, 10)}"
        sup_title = f"Logarithmic Loss (min. {min_text})"
        title = titles[o]
    return c_map, sup_title, title, v_min, v_max


def print_expected_output(U, x, name):
    """convience function to help print expected outputs of a unitary and input data points

    Args:
        U (tensor): a simple unitary matrix in tensor form
        x (tensor): data points
        name (string): name of the unitary/type of data
    """
    print("====")
    expected_output = torch.matmul(U, x)
    np_arr = expected_output.detach().cpu().numpy()
    print("expected output for ", name, ":\n", np_arr)
    print("====")


def print_datapoints(points, title):
    """a little helper to print data points to console more conveniently

    Args:
        points (torch tensor): tensor containing the data used to train a qnn
        title (string): describes what kind of data points (i.e. entangled..)
    """
    print("", title, " data points:")
    np_arr = points.detach().cpu().numpy()
    for i, row in enumerate(np_arr):
        print("---")
        for j, point in enumerate(row):
            print("", i, " - ", j, ":", point)


def get_k_norm(arr, k):
    """this function calculates the entry wise k-norm for an n-dimensional array https://en.wikipedia.org/wiki/Matrix_norm

    Args:
        arr (array): n-dimensional array
        k (int > 0): indicator of which norm you want to use (i.e. 1-norm, 2-norm, ...)

    Returns:
        int: the k-norm
    """
    arr = np.array(arr)
    inner_sum = 0
    for num in np.nditer(arr):
        inner_sum += np.absolute(num) ** k
    return inner_sum ** (1.0 / k)


def get_first_order_gradient_of_point(derivative_direction, target_point, landscape):
    """given an n dimensional array, this calculates the first oder gradient of a single point in a single direction

    Args:
        derivative_direction (int): id of the direction of the gradient
        target_point (list): the point you want to calculate the gradient of
        landscape (array): landscape in which the point can be found

    Returns:
        float: the first order gradient of the target point in the landscape in the target direction
    """
    grid_size = len(landscape[0])
    if target_point[derivative_direction]==0:
        #forward diff
        leftid = list(target_point)
        leftid[derivative_direction] = leftid[derivative_direction]+1
        rightid = list(target_point)    
        leftidx = tuple(leftid)
        rightidx = tuple(rightid)
        return (landscape[leftidx]-landscape[rightidx])
    if target_point[derivative_direction] >= grid_size-1:
        #backward diff        
        leftid = list(target_point)
        rightid = list(target_point)
        rightid[derivative_direction] = rightid[derivative_direction] -1
        
        leftidx = tuple(leftid)
        rightidx = tuple(rightid)
        return (landscape[leftidx]-landscape[rightidx])
    
    leftid = list(target_point)
    rightid = list(target_point)
    leftid[derivative_direction] = leftid[derivative_direction]+1
    rightid[derivative_direction] = rightid[derivative_direction] -1
    leftidx = tuple(leftid)
    rightidx = tuple(rightid)
    return (landscape[leftidx]-landscape[rightidx])/2

def get_second_order_gradient_of_point(first_order_direction,second_order_direction, target_point, landscape):
    """calculates the second order derivative in the specified directions

    Args:
        first_order_direction (int): the id of the direction of the first order derivative
        second_order_direction (int): the id of the direction of the second order derivative
        target_point (list): the point in the landscape to derive
        landscape (array): loss landscape

    Returns:
        float: the second order derivative at the point of the landscape specified
    """
    grid_size = len(landscape[0])
    if target_point[second_order_direction]==0:
        #forward diff
        leftid = list(target_point)
        leftid[second_order_direction] = leftid[second_order_direction]+1
        rightid = list(target_point)    
        leftidx = tuple(leftid)
        rightidx = tuple(rightid)
        return (get_first_order_gradient_of_point(first_order_direction, leftidx, landscape)-get_first_order_gradient_of_point(first_order_direction, rightidx, landscape))
    if target_point[second_order_direction] >= grid_size-1:
        #backward diff        
        leftid = list(target_point)
        rightid = list(target_point)
        rightid[second_order_direction] = rightid[second_order_direction] -1        
        leftidx = tuple(leftid)
        rightidx = tuple(rightid)
        return (get_first_order_gradient_of_point(first_order_direction, leftidx, landscape)-get_first_order_gradient_of_point(first_order_direction, rightidx, landscape))    
    leftid = list(target_point)
    rightid = list(target_point)
    leftid[second_order_direction] = leftid[second_order_direction]+1
    rightid[second_order_direction] = rightid[second_order_direction] -1
    leftidx = tuple(leftid)
    rightidx = tuple(rightid)
    return (get_first_order_gradient_of_point(first_order_direction, leftidx, landscape)-get_first_order_gradient_of_point(first_order_direction, rightidx, landscape))/2
