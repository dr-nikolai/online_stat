'''
Functions for testing online (single-pass) statistics objects.

Author: Nikolai Shokhirev, http://www.numericalexpert.com/
Date: December, 26, 2017
'''

import numpy as np

def win_moment(X, n, imax, p):
    '''
    Windowed mean and central moments
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

def cum_moment(X, imax, p):
    '''
    Cumulative mean and central moments
    Two-pass algorithm - used for testing
    X - data array
    imax - the highest index
    p - power
    if p == 1: 
        S - sum of last n available x terms
        M - mean of last n available terms
    if p > 1:
        S - sum of last n available (x-mean)**p
        M - central p-moment of last n available terms
    '''
    M = win_moment(X, imax+1, imax, p)
    return M

def win_cov(X, Y, n, imax):
    '''
    Windowed covariance two-pass algorithm (For testing)
    n - target window
    imax - the highest index
        S - sum of last n available (x-x.mean)*(y-y.mean)
        C - covariance of last n available terms
    '''
    l = max(0, imax-n+1)
    h = min(imax+1,len(X))
    x = X[l:h]
    y = Y[l:h]
    mx = x.mean()
    my = y.mean()
    S = np.sum((x-mx)*(y-my))
    C = S/(h-l)
    return C
        
def cum_cov(X, Y, imax):
    '''
    Cumulative covariance two-pass algorithm (For testing)
    imax - the highest index
        S - sum of last n available (x-x.mean)*(y-y.mean)
        C - covariance of last n available terms
    '''
    C = win_cov(X, Y, imax+1, imax)
    return C