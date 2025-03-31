import time
import numpy as np
import scipy as sp
from miser.MISER import *
from miser.SimpleMonteCarlo import *
from miser.test_functions import *
from BA_testing import rosen_projection_to_2d
from metrics import calc_scalar_curvature_for_function
from BA_grid_TASC import get_basic_3D_cost_function

def test_miser():
    '''
        random first test, to understand how it works
    '''
    # batman
    low = [-7, -3]
    high = [7, 3]
    ave, var, pts = mc_integrate(batman, low, high, int(5e4))
    pts = get_points(pts)
    fig, ax = plt.subplots(figsize=(9,5))
    plt.xlim([low[0], high[0]])
    plt.ylim([low[1], high[1]])
    patches = []

    # for t in miser.terminals:
    #     x = t[0]
    #     y = t[1]
    #     width = t[2] - t[0]
    #     height = t[3] - t[1]
    #     patches.append(Rectangle((x,y), width, height, facecolor=(0,0,0,0), edgecolor='#eeeeee'))

    # batman curve
    x = np.linspace(low[0], high[0], 1000)
    batman_up = [batman_upper(pt) for pt in x]
    batman_low = [batman_lower(pt) for pt in x]

    ax.fill_between(x, batman_low, batman_up, facecolor='#fde311')
    ax.fill_between(x, batman_up, 7*np.ones(1000), facecolor='#000000')
    ax.fill_between(x, -7*np.ones(1000), batman_low, facecolor='#000000')

    # points
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, batman_up, 'k-', linewidth=2)
    plt.plot(x, batman_low, 'k-', linewidth=2)
    plt.plot(pts[0], pts[1], 'r,')

    plt.show()

    # TESTING - multidimensional simple Monte Carlo
    # the test function
    func = lambda x: sum(x)**2
    # the region
    low = np.zeros(10)
    high = np.ones(10)
    # values of N to test with
    N = [10**i for i in range(3, 4)] # changed from range(3,8) to range(3,6)
    Nlen = len(N)
    sintegral = np.zeros(Nlen)
    serror = np.zeros(Nlen)
    for i in range(Nlen):
        print('N = {:.4g}'.format(N[i]))
        sintegral[i], serror[i], pts = mc_integrate(func, low, high, N[i])

    R = 1
    pi, pts = estimate_pi(R, 10**6)
    fig, ax = plt.subplots(figsize=(5,5))
    plt.xlim([-R-0.00, R+0.00])
    plt.ylim([-R-0.00, R+0.00])
    x1 = np.linspace(-R, R, 1000)
    y1 = np.sqrt(R**2 - x1**2)
    y2 = -1*y1
    plt.plot(x1, y1, "k-", lw=2)
    plt.plot(x1, y2, "k-", lw=2)
    inside_pts = list(filter(lambda x: x[0]**2 + x[1]**2 < R**2, pts))
    inx = []
    iny = []
    for pt in inside_pts:
        inx.append(pt[0])
        iny.append(pt[1])
    plt.plot(pts[:,0], pts[:,1], 'b,')
    plt.plot(inx, iny, 'r,')
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.show()

    save, svar, points = mc_integrate(hyperbatman, [-7, -3, -1], [7, 3, 1], 10**3)

    ns = [i for i in range(3, 4)] # changed from range(3,11) to range(3,6)
    sactual = []
    scomputed = []
    serror = []
    srel = []
    j = 0
    for n in ns:
        print('n =', n)
        low = [-7, -3]
        high = [7, 3]
        for i in range(2, n):
            low.append(-1)
            high.append(1)
        ave, var, p = mc_integrate(hyperbatman, low, high, 10**3)
        scomputed.append(ave)
        sactual.append(batman_area*2**(n-2))
        srel.append(abs(sactual[j]-scomputed[j])/abs(sactual[j]))
        serror.append(var)
        j += 1

    Ns = [10**i for i in range(3, 4)] # changed from range(3,8) to range(3,6)
    V = 14*7*2**6
    actual = batman_area*2**6
    low = [-7, -3, -1, -1, -1, -1, -1, -1]
    high = [7, 3, 1, 1, 1, 1, 1, 1]
    scomputed = []
    serror = []
    srel = []
    for N in Ns:
        print('N =', N)
        ave, var, p = mc_integrate(hyperbatman, low, high, N)
        scomputed.append(ave)
        serror.append(var)
        srel.append(abs(actual-ave)/abs(actual))

def integrate_cube_MC(func, lowerleft, upperright, N=1000):
    '''
        Assumption: cube and not rectangular box, i.e. side lengths are the same for every dimension
    '''
    start = time.time()
    dim = len(lowerleft)
    if dim != len(upperright):
        raise Exception("wrong dimensions")
    volume = 1
    for i in range(dim):
        side = upperright[i]-lowerleft[i]
        volume *= side
    sample_points = []
    rng = np.random.default_rng()
    sample_points = rng.uniform(low = lowerleft, high = upperright, size = (N, dim))
    fun_values = []
    for point in sample_points:
        fun_values.append(func(point))
    result = np.mean(fun_values)*volume
    elapsed_time = time.time()-start
    return result, np.round(elapsed_time, 3)

def integrate_cube_MISER(func, lowerleft, upperright, N=1000):
    start = time.time()
    miser_instance = MISER()
    dim = len(lowerleft)
    volume = 1
    for i in range(dim):
        side = upperright[i]-lowerleft[i]
        volume *= side
    ave, var, N = miser_instance.MISER(func,lowerleft, upperright, N=N, dith=0)
    #miser_result = ave*volume
    elapsed_time = time.time()-start
    return ave, N, np.round(elapsed_time, 3)

def get_ASC_function(func):
    def absolute_scalar_curvature(x):
        points = [x.tolist()]
        result = calc_scalar_curvature_for_function(rosen_projection_to_2d, points)
        return np.absolute(result[0])
    return absolute_scalar_curvature

def compare_Miser_with_MC_TASC(func, lowerleft, upperright, N=1000):
    '''
        Comparison of TASC of function func within a hypercube when intergral is computed with regular Monte Carlo
        or MISER. Hypercube is given by a lower left corner and upper right corner.
    '''
    absolute_SC = get_ASC_function(func)
    mc_results = []
    times_mc = []
    miser_results = []
    times_miser = []
    Ns_miser = []
    for _ in range(100):
        mc_result, time_mc = integrate_cube_MC(absolute_SC, lowerleft, upperright, N=N)
        miser_result, N_miser, time_miser = integrate_cube_MISER(absolute_SC, lowerleft, upperright, N=N)
        mc_results.append(mc_result)
        times_mc.append(time_mc)
        miser_results.append(miser_result)
        times_miser.append(time_miser)
        Ns_miser.append(N_miser)
    print(f"Total absolute SC of function {func}.")
    print(f"Integrated within hypercupe with lowerleft={lowerleft}, upperright={upperright}")
    print(f"Median results over 100 repetitions")
    print(f"regular MC: TASC = {np.median(mc_results)}, N={N}, time (s)= {np.median(times_mc)}")
    print(f"MISER: TASC={np.median(miser_results)}, N={np.median(Ns_miser)}, time (s)= {np.median(times_miser)}")


if __name__=="__main__":
    Ns = [100,500,1000]
    func = get_basic_3D_cost_function()
    for N in Ns:
        compare_Miser_with_MC_TASC(func, [0,0,0], [2,2,2], N=N)