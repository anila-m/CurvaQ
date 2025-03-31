import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_2D_surface(points, values):
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
    ax.plot_surface(X, Y, values, rstride=1, cstride=1, cmap=cm.viridis, linewidth=0.1)
    ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel="values")
    plt.title("")
    plt.show()


def calc_landscape_tasc(grid_point_array, r=0):
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
    landscape = np.empty(landscape_shape)
    j = 0
    for idx, _ in np.ndenumerate(landscape):
        point = []
        # generate point in grid array
        i=0
        for dimension in idx:
            point.append(grid_point_array[i][dimension])
            i += 1
        point = np.asarray(point)
        landscape[idx] = j
        print("idx", idx, "j", j, "point", point)

if __name__=="__main__":
    low = [1,2,1]
    high = [4,4,4]
    n = len(low)
    rng = np.random.default_rng()
    s = rng.uniform(low, high, size=n)
    print(s)
