import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

from matplotlib import cm
from BA_grid_TASC import calc_landscape_tasc, get_array_summary, plot_2D_overlapping_circles, generate_grid_point_array
from metrics import *
from BA_testing_functions import cosine_2D, get_ASC_function
from BA_miser_test import integrate_cube_MC

# Various plots for thesis that don't illustrate experiment results, e.g. of test functions
# used in Chapters 2 (Theoretical Foundations), 5 (Curvature), 6 (Implementation)

def plot_cosine_2D():
    '''
        Creates plot of function values and calculalates TASC and TSC of f inside [-pi/2,pi/2]^2 cube.
        With f(x1,x2) = 1/4(cos(4x1)-xos(4x2)).
    '''
    stepsize = np.pi/2
    c = np.asarray([0,0])
    ll = [-stepsize, -stepsize]
    ur = [stepsize, stepsize]
    print(ll, ur)
    absolute_SC = get_ASC_function(cosine_2D)
    SC = get_ASC_function(cosine_2D, absolute=False)
    tasc = 0
    tsc = 0
    N=100000
    for i in range(10):
        tasc += integrate_cube_MC(absolute_SC, ll, ur, N=N)[0]
        tsc += integrate_cube_MC(SC, ll, ur, N=N)[0]
    print(tasc/10, tsc/10)
    filepath = "plots/preliminary_tests/cosine_2D_functionValues"
    x = np.linspace(-stepsize,stepsize,100)
    X,Y = np.meshgrid(x,x)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    values = []
    for x1 in x:
        for x2 in x:
            values.append(cosine_2D([x1,x2]))
    values = np.reshape(np.asarray(values), X.shape)
    ax.plot_surface(Y, X, values, rstride=1, cstride=1, cmap=cm.viridis, linewidth=0.1) #switch Y and X, since that's how our TASC value landscapes are computed
    ax.set_xlabel(xlabel='$x$', fontsize=14)
    ax.set_ylabel(ylabel='$y$', fontsize=14)
    #ax.set_zlabel(zlabel="$f(x_1,x_2)$", fontsize=14)
    ticks = [-stepsize, -stepsize/2, 0, stepsize/2, stepsize]
    ticklabel = ["$-\pi/2$","$-\pi/4$","0","$\pi/4$","$\pi/2$"]
    ax.set_xticks(ticks, labels=ticklabel, fontsize=12)
    ax.set_yticks(ticks, labels=ticklabel, fontsize=12)
    ax.tick_params(axis='z', which='major', labelsize=12)
    #plt.title("$f(x_1,x_2) = 1/2(cos(2x_1)+cos(2x_2))$", fontsize=16)
    plt.tight_layout(pad=0.6, w_pad=0.5, h_pad=1.0)
    #plt.savefig(f"{filepath}.png", dpi=dpi_number, bbox_inches = "tight")
    plt.savefig(f"{filepath}.pdf", format='pdf', bbox_inches = "tight")
    plt.close()

def cosine_2D_critical_points_plot():
    """
        Critical points of f inside [-pi/2,pi/2]^2 cube.
        With f(x1,x2) = 1/4(cos(4x1)-xos(4x2)).
    """
    minima_x, minima_y = [], []
    maxima_x, maxima_y = [], []
    saddle_x, saddle_y = [], []
    for i in range(3):
        for j in range(3):
            point_x = i*np.pi/4
            point_y = j*np.pi/4
            if i%2==0 and j%2 == 0:
                maxima_x.extend([point_x, -point_x, point_x, -point_x])
                maxima_y.extend([point_y, point_y, -point_y, -point_y])
            elif i%2==1 and j%2==1:
                minima_x.extend([point_x, -point_x, point_x, -point_x])
                minima_y.extend([point_y, point_y, -point_y, -point_y])
            else:
                saddle_x.extend([point_x, -point_x, point_x, -point_x])
                saddle_y.extend([point_y, point_y, -point_y, -point_y])
    # determine SC values 
    grid_size = 100 # in total 1000x1000 points on regular grid
    x = np.linspace(-np.pi/2,np.pi/2,grid_size)
    X,Y = np.meshgrid(x,x)
    points = np.ndarray((grid_size**2,2))
    points[:,0] = X.flatten()
    points[:,1] = Y.flatten()
    sc_values,_,_ = calc_scalar_curvature_for_function(cosine_2D,points)
    # plot SC as contour plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    plt.contourf(Y, X, sc_values.reshape(X.shape), cmap=cm.viridis)
    cbar = plt.colorbar()
    cbar.set_label('Scalar Curvature', rotation=90, fontsize=18)
    cbar.ax.tick_params(labelsize=16)
    # plot critical points
    ticks = [-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2]
    ticklabel = ["$-\pi/2$","$-\pi/4$","0","$\pi/4$","$-\pi/2$"]
    ax.set_xticks(ticks,labels=ticklabel, fontsize=16)
    ax.set_yticks(ticks,labels=ticklabel, fontsize=16)
    plt.xlabel('$x$', fontsize=14)
    plt.ylabel('$y$', fontsize=14)

    min = plt.scatter(minima_x, minima_y, facecolors='none', edgecolors='black', marker='o')
    max = plt.scatter(maxima_x, maxima_y, color='black', marker='o')
    sad = plt.scatter(saddle_x, saddle_y, color='black', marker='x')
    plt.legend((min, max, sad),
           ('Minimum', 'Maximum', 'Saddle point'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=13)
    plt.tight_layout()
    #plt.savefig("plots/preliminary_tests/cosine_2D_criticalPointsSC.png",dpi=dpi_number)
    plt.savefig("plots/preliminary_tests/cosine_2D_criticalPointsSC.pdf", format='pdf')
    
def cosine_2D_overlappingCircles_tasc():
    '''
        Creates a plot with tasc values for overlapping circles in the 2D cube [-pi/2, pi/2]^2.
        number of circles in each direction: 9.
        centers of circles:[k*pi/8, l*pi/8] for k,l \in [-4, -3, ...., 3, 4], radius: pi/8
    '''
    points = np.linspace(-np.pi/2,np.pi/2,9)
    grid_point_array = generate_grid_point_array(np.pi/8,[-np.pi/2, -np.pi/2],9)
    tasc_landscape,_,_,_ = calc_landscape_tasc(cosine_2D,grid_point_array, no_samples=10000)
    filename = "cosine_2D_overlappingCircles_smallerCube_10000_withCriticalPoints"
    label = "TASC"
    ticklabels = ["$-\pi/2$", "$-\pi/4$", "$0$", "$\pi/4$", "$\pi/2$"]
    ticks = [-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2]
    plot_2D_overlapping_circles(grid_point_array, tasc_landscape, label, "", filename, ticklabels=ticklabels,ticks=ticks, cosine=True)
    print("TASC landsacpe summary")
    get_array_summary(tasc_landscape,printSummary=True)

def make_f_plots_for_thesis():
    """
        Several plots for f(x1,x2) = 1/4(cos(4x1)-xos(4x2)) for thesis. (Chapter Curvature)
    """
    # plot of overlapping circles with encoded TASC values for each circle (colormap) covering [-pi/2,pi/2]^2 cube
    cosine_2D_overlappingCircles_tasc()

    # TASC and TSC value inside [-pi/2,pi/2]^2 cube
    # and plot of function values saved as "plots/preliminary_tests/cosine_2D_functionValues.png"
    plot_cosine_2D()

    # plot of critical points of f inside [-pi/2,pi/2]^2 cube
    cosine_2D_critical_points_plot()

def plot_basic_loss_landscape():
    '''
        Plot for VQA diagram in thesis. (Chapter 2 Theoretical Foundations)
    '''
    def f(x,y):
        return 1+x*sin(y)+0.5*y*sin(x)

    stepsize=5
    filepath = "plots/preliminary_tests/basic_2d_costFunction"
    x = np.linspace(-stepsize,stepsize,100)
    X,Y = np.meshgrid(x,x)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    values = []
    for x1 in x:
        for x2 in x:
            values.append(f(x1,x2))
    values = np.reshape(np.asarray(values), X.shape)
    ax.plot_surface(Y, X, values, rstride=1, cstride=1, cmap=cm.viridis, linewidth=0.1) #switch Y and X, since that's how our TASC value landscapes are computed
    #plt.title("$f(x_1,x_2) = 1/2(cos(2x_1)+cos(2x_2))$", fontsize=16)
    plt.tight_layout(pad=0.6, w_pad=0.5, h_pad=1.0)
    ax.grid(False)
    plt.xticks([])
    plt.yticks([])
    ax.set_zticks([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    plt.savefig(f"{filepath}.pdf", bbox_inches = "tight", transparent=True)
    plt.close()

# Helper Functions for plots

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
    plt.savefig(f"plots/preliminary_tests/{file_name}.png", format='pdf')
    plt.close()

def plot_2D_overlapping_circles(points, values, label, title, file_name, ticklabels=[], ticks=[], cosine=False):
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

    #determine radius of circles
    r = points[0,1]-points[0,0]

    # Plot circles
    plt.figure(figsize=(8,6))
    plt.grid()
    ax = plt.gca()
    ax.set_axisbelow(True)
    for a, b, color in zip(df['X'], df['Y'], colors):
        circle = plt.Circle((a, 
                            b), 
                            radius=r, # radius
                            color=color, 
                            fill=True,
                            alpha=0.5)
        ax.add_artist(circle)

    plt.xlim([np.min(points[0])-r,np.max(points[0])+r])
    plt.ylim([np.min(points[1])-r,np.max(points[1])+r])
    if len(ticks)>0 and len(ticklabels)>0:
        plt.xticks(ticks=ticks, labels=ticklabels)
        plt.yticks(ticks=ticks, labels=ticklabels)
    #plt.xlabel('$x$', fontsize=14)
    #plt.ylabel('$y$', fontsize=14)
    ax.set_aspect(1.0)

    sc = plt.scatter(df['X'], df['Y'], s=0, c=c1, cmap='viridis', facecolors='none')

    cbar = plt.colorbar(sc)
    cbar.set_label(label, labelpad=10, fontsize=14)

    # if the values belong to the cosine_2D function used for thesis (as an example): also plot critical points
    if cosine:
        minima_x, minima_y = [], []
        maxima_x, maxima_y = [], []
        saddle_x, saddle_y = [], []
        for i in range(3):
            for j in range(3):
                point_x = i*np.pi/4
                point_y = j*np.pi/4
                if i%2==0 and j%2 == 0:
                    maxima_x.extend([point_x, -point_x, point_x, -point_x])
                    maxima_y.extend([point_y, point_y, -point_y, -point_y])
                elif i%2==1 and j%2==1:
                    minima_x.extend([point_x, -point_x, point_x, -point_x])
                    minima_y.extend([point_y, point_y, -point_y, -point_y])
                else:
                    saddle_x.extend([point_x, -point_x, point_x, -point_x])
                    saddle_y.extend([point_y, point_y, -point_y, -point_y])
        min = plt.scatter(minima_x, minima_y, facecolors='none', edgecolors='black', marker='o')
        max = plt.scatter(maxima_x, maxima_y, color='black', marker='o')
        sad = plt.scatter(saddle_x, saddle_y, color='black', marker='x')
        plt.legend((min, max, sad),
            ('Minimum', 'Maximum', 'Saddle point'),
            scatterpoints=1,
            loc='lower left',
            ncol=3,
            fontsize=10)
    if title != "":
        plt.title(title, fontsize=16)
    #plt.savefig(f"plots/preliminary_tests/{file_name}.png", format='pdf')
    plt.savefig(f"plots/preliminary_tests/{file_name}_noLabels.pdf", format='pdf')
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
    plt.savefig(f"plots/preliminary_tests/{file_name}.png", format='pdf')
    plt.close()

if __name__=="__main__":

    make_f_plots_for_thesis()


