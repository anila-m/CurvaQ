import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm

from metrics import *

def f(x):
    if x.shape[0] != 2:
        raise Exception("Input has to have length of 2.")
    return x[0]**2 - x[1]**2

def plot_function(func, point, radius):
    grid_size = 100
    x = np.linspace(point[0]-radius,point[0]+radius,grid_size)
    y = np.linspace(point[1]-radius,point[1]+radius,grid_size)
    X,Y = np.meshgrid(x,y)
    points = np.ndarray((grid_size**2,2))
    points[:,0] = X.flatten()
    points[:,1] = Y.flatten()
    sc_values = calc_scalar_curvature_for_function(f,points).reshape(X.shape)
    ax = plt.subplot(111, projection='3d')
    ax.plot_surface(X, Y, sc_values, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='SC')
    plt.title("Scalar Curvature for $f(x,y) = x^2 - y^2$")
    plt.savefig("plots/preliminary_tests/f_SC_Values.png", dpi=500)
    plt.close()
    del ax
    func_values = np.ndarray(points.shape[0])
    for idx in range(points.shape[0]):
        func_values[idx] = f(points[idx])
    ax = plt.subplot(projection='3d')
    ax.plot_surface(X, Y, func_values.reshape(X.shape), cmap=cm.coolwarm,linewidth=0, antialiased=False)
    ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$f(x_1,x_2)$')
    plt.title("Function values for $f(x,y) = x^2 - y^2$")
    plt.savefig("plots/preliminary_tests/f_func_Values.png", dpi=500)
    plt.close()

if __name__ == "__main__":
    c = np.asarray([0,0])
    r = 5
    plot_function(f, c, r)