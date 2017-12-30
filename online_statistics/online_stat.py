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
        self.reset(n)
        
    def reset(self, n):
        """
            Can be called to reuse existing object
        """
        self.s = 0.0
        self.s2 = 0.0
        self.s3 = 0.0
        self.s4 = 0.0
        self.k = 0
        self.m = 0.0       
        self.xn = np.zeros(n)
        
    def push(self, x):
        """
            Update object with new data x 
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
        
        
class CumulativeStat():
    """
        Cumulative  statistics
        Nikolai Shokhirev, 2017
        see Single-Pass Online Statistics Algorithms, 2013
        http://www.numericalexpert.com/articles/single_pass_stat/ 
        Calculation of m (mean), and m2, m3 and m4 centered moments
    """
    def __init__(self):
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
            Update object with new data x
        """
        m1 = self.m
        d = x - m1
        self.k += 1.0
        self.s += x
        self.s4 += d**4 - 4.0*d *(self.s3 + d**3) / self.k \
              + 6.0*(self.s2 + d**2)*d**2 / self.k**2 - 3.0*d**4 / self.k**3
        self.s3 += d**3 - 3.0*d*(self.s2 + d**2) / self.k + 2.0*d**3 / self.k**2
        self.s2 += d**2*(1.0 - 1.0 / self.k)
        self.m = self.s / self.k
        self.m2 = self.s2/self.k
        self.m3 = self.s3/self.k
        self.m4 = self.s4/self.k

    
class WindowedCovariance():
    """
        Running windowed covariance
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
        self.reset(n)
        
    def reset(self, n):
        """
            Can be called to reuse existing object
        """
        self.sx = 0.0
        self.sy = 0.0
        self.sxy = 0.0
        self.k = 0
        self.xn = np.zeros(n)
        self.yn = np.zeros(n)
        self.mx = 0.0
        self.my = 0.0
        
    def push(self, x, y):
        """
            Update object with new data x 
        """        
        mx1 = self.mx
        my1 = self.my
        dx = x - mx1
        dy = y - my1
        if self.k < self.n:
            self.k += 1.0
            self.sx += x
            self.sy += y
            dxn = 0.0
            dyn = 0.0
        else:
            self.sx += x - self.xn[0]
            dxn = self.xn[0] - mx1 
            self.sy += y - self.yn[0]
            dyn = self.yn[0] - my1 
        self.sxy += dx*dy - dxn*dyn - (dx - dxn)*(dy - dyn)/self.k
        self.xn = np.roll(self.xn,-1)
        self.yn = np.roll(self.yn,-1)
        self.xn[-1] = x
        self.yn[-1] = y
        self.mx = self.sx/self.k
        self.my = self.sy/self.k
        self.cov = self.sxy/self.k
        
class CumulativeCovariance():
    """
        Running windowed covariance
        Nikolai Shokhirev, 2017
        see Single-Pass Online Statistics Algorithms, 2013
        http://www.numericalexpert.com/articles/single_pass_stat/ 
        Calculation of m (mean), and m2, m3 and m4 centered moments
    """

    def __init__(self):
        """
            n - target window size
        """
        self.reset()
        
    def reset(self):
        """
            Can be called to reuse existing object
        """
        self.sx = 0.0
        self.sy = 0.0
        self.sxy = 0.0
        self.k = 0
        self.mx = 0.0
        self.my = 0.0
        
    def push(self, x, y):
        """
            Update object with new data x 
        """        
        mx1 = self.mx
        my1 = self.my
        dx = x - mx1
        dy = y - my1
        self.k += 1.0
        self.sx += x
        self.sy += y
        self.sxy += dx*dy - dx*dy/self.k
        self.mx = self.sx/self.k
        self.my = self.sy/self.k
        self.cov = self.sxy/self.k