'''
Suppliment to the article 
Nikolai Shokhirev, "Single-Pass Online Statistics Algorithms", 2013
http://www.numericalexpert.com/articles/single_pass_stat/

Author: Nikolai Shokhirev, http://www.numericalexpert.com/
Date: December, 26, 2017
'''

import numpy as np

class WindowedStat():
    """
        Windowed statistics
        Nikolai Shokhirev, 2017
        see Single-Pass Online Statistics Algorithms, 2013
        http://www.numericalexpert.com/articles/single_pass_stat/ 
        Calculation of m (mean), and m2, m3 and m4 centered moments
    """
    def __init__(self,n):
        """
            n - target window size
        """
        self.n = n
        self.xn = np.zeros(n)
        self.reset()
        
    def reset(self):
        """
            Can be called to reuse existing object
        """
        self.s = 0.0
        self.s2 = 0.0
        self.s3 = 0.0
        self.s4 = 0.0
        self.k = 0
        self.m = 0.0       
        
    def push(self, x):
        """
            Update object with new data
            x - data 
        """
        m1 = self.m
        d = x - m1
        if self.k < self.n: # Initial cumulativestatistics.
            self.k += 1.0
            self.s += x
            dn = 0.0
        else:               # Windowed statistics.
            self.s += x - self.xn[0]
            dn = self.xn[0] - m1
        self.s4 += d**4 - dn**4 - 4.0*(d - dn)*(self.s3 + d**3 - dn**3) / self.k \
              + 6.0*(self.s2 + d**2 - dn**2)*(d - dn)**2 / self.k**2 - 3.0*(d - dn)**4 / self.k**3
        self.s3 += d**3 - dn**3 - 3.0*(d - dn)*(self.s2 + d**2 - dn**2) / self.k \
                   + 2.0*(d - dn)**3 / self.k**2
        self.s2 += d**2 - dn**2 - (d - dn)**2 / self.k
        self.xn = np.roll(self.xn, -1) # Shift array values to the left
        self.xn[-1] = x                # adding new value to the end of xn
        self.m = self.s / self.k
        self.m2 = self.s2/self.k
        self.m3 = self.s3/self.k
        self.m4 = self.s4/self.k
        
        
def moment(X, n, imax, p):
    '''
    Two-pass algorithm - used for testing
    X - data array
    n - target window
    imax - the highest index
    p - power
    if p == 1: 
        S - sum of last n available x terms
        M - mean of last n available terms
    if p > 1:
        S - sum of last n available (x-mean)**p
        M - central p-moment of last n available terms
    '''
    l = max(0, imax-n+1)
    h = min(imax+1,len(X))
    x = X[l:h]
    m = x.mean() if p > 1 else 0.0
    S = np.sum((x-m)**p) 
    M = S/(h-l)
    return M
        
        
        