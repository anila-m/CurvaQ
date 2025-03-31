import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
from metrics import *

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

def cosine_2D(x):
    if len(x) != 2:
        raise Exception
    Z = (1/2) * (cos(2 * x[0]) + cos(2 * x[1]))
    return Z

def cosine_2D_tasc():
    stepsize = np.pi*2
    c = np.asarray([0,0])
    tascs = []
    tscs = []
    mascs = []
    mscs = []
    for _ in range(100):
        tasc, tsc, masc, msc = calc_several_scalar_curvature_values(cosine_2D, stepsize, c)
        tascs.append(tasc)
        tscs.append(tsc)
        mascs.append(masc)
        mscs.append(msc)
    print("tasc", np.mean(tascs))
    print("tsc", np.mean(tscs))
    print("masc", np.mean(mascs))
    print("msc", np.mean(mscs))
    filepath = "plots/preliminary_tests/cosine_2D_v2.png"
    x = np.linspace(-stepsize,stepsize,500)
    X,Y = np.meshgrid(x,x)
    ax = plt.subplot(111, projection='3d')
    values = []
    for x1 in x:
        for x2 in x:
            values.append(cosine_2D([x1,x2]))
    values = np.reshape(np.asarray(values), X.shape)
    ax.plot_surface(Y, X, values, rstride=1, cstride=1, cmap=cm.viridis, linewidth=0.1) #switch Y and X, since that's how our TASC value landscapes are computed
    ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel="$f(x_1,x_2)$")
    plt.title("$f(x_1,x_2) = 1/2(cos(2x_1)+cos(2x_2))$")
    plt.savefig(filepath, dpi=500)
    plt.close()

def analyze_gradients_high_tasc():
    points = []


if __name__=="__main__":
    cosine_2D_tasc()
