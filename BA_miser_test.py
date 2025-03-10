import numpy as np
import scipy as sp
from miser.MISER import *
from miser.SimpleMonteCarlo import *
from miser.test_functions import *

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

if __name__=="__main__":
    test_miser()