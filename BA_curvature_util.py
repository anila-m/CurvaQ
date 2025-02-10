import numpy as np
import scipy as sp

from metrics import *

rng = np.random.default_rng()

# Volume of n dimensional hypersphere
# Alina
def get_hypersphere_volume(n, r):
    '''
    Computes volume of a n-dimensional hypersphere (n-ball) with radius r.

    Args:
        n (int): number of dimensions
        r (float): radius of hypersphere
    
    Returns:
        scalar: volume of hypersphere
    '''
    volume = (r**n)*((np.pi**(n/2))/sp.special.gamma(n/2+1))
    return volume

# helper function: sampling points inside n-ball (uniformly random)
# Überlegung: "normale" Methode vermutlich minimal schneller, da Dimension n statt n+2
# Alina
def sample_n_ball_uniform(n, r, c, N):
    '''
    Sampling N points inside an n-ball with center c and Radius r using the Box-Muller-Transform.

    Step 1: Sample point x=(x_1,...,x_n) on the (n-1)-sphere with center (0,...,0) (dimension: n) 
            using the Marsaglia method, based on Box-Muller-Transform
    Step 2: Uniformly sample u from [0,1]
    Step 3: Compute r*u^(1/n)*x + c

    Args: 
        n (int): dimension
        r (float): radius of ball
        c (array): center of ball, shape: 1xn
        N (int): number of sample points to be generated
    
    Results:
        sample_points (array): N sample points, shape: Nxn
    '''
    # Step 1: Sample points on (n-1) sphere and add center c (shift to unit sphere with center c)
    sample_points = sample_unit_n_sphere_uniform(n-1,N)
    #print(sample_points)
    # Step 2 + 3: sample u from [0,1] for each point x and compute r * u^(1/n) * x
    u = rng.uniform(low=0,high=1,size=(N,1))
    #print(u)
    sample_points = sample_points * (u**(1/n))
    #print(sample_points)
    sample_points *= r
    sample_points += c
    return sample_points

def sample_unit_n_sphere_uniform(n,N):
    '''
    Sample N points on the unit n-sphere (in R^(n+1)), i.e. the sphere with center (0,...,0) in R^(n+1) and radius 1.
    Source: Voelker, et al (2017): Efficiently sampling vectors and coordinates from the n-sphere and n-ball (section 2.1)

    Args:
        n (int): dimension
        N (int): number of points to be sampled

    Results:
        sample_points (array): array of sampled points on the unit n-sphere, shape: Nx(n+1)
    '''
    # sample Nx(n+1) normally distributed values between -1 and 1 (mean is 0, standard deviation is 1)
    sample_points = rng.normal(0,1,(N,n+1))
    # compute l2-Norm of each point (i.e. each row) and divide each point by its l2-Norm
    sample_points /= np.linalg.norm(sample_points,axis=1,keepdims=True)
    return sample_points
    

# helper function: sampling points inside n-ball (uniformly random) using Voelker method
# eventuell nicht implementieren, "normale" Methode reicht
# Alina
def sample_n_ball_uniform_Voelker_method(n, r, c, N):
    '''
    Sampling N points inside an n-ball with center c and Radius r using the method introduced in 
    Voelker, et al (2017): Efficiently sampling vectors and coordinates from the n-sphere and n-ball

    Step 1: Sample points x=(x_1,...,x_n+2) on the (n+1)-sphere with center c (dimension: n+2)
    Step 2: Discard two coordinates (dimension: n) 

    Args: 
        n (int): dimension
        r (float): radius of ball
        c (array): center of ball, shape: 1xn
        N (int): number of sample points to be generated
    
    Results:
        sample_points (array): N sample points, shape: Nxn
    '''
    # TODO: Implementierung
    sample_points = []
    return sample_points