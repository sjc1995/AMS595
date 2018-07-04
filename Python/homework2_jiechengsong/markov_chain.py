#!/usr/bin/python
#Author: Jiecheng Song SBID:111783762

import numpy as np

def markov_chain(N):
    '''define a function called mandelbrot_1 with parameter N'''
    P = np.random.rand(N,N)
    '''make a NxN random matrix P'''
    P_sum = P.sum(axis=1)
    '''sum the P by row and get P_sum'''
    for i in range(N):
        P[i,:] = P[i,:]/P_sum[i]
    '''normalize the P, let the sum of each row become 1'''
    p = np.random.rand(N,1)
    '''generate a rondom vector p'''
    p = p/(p.sum(axis=0))
    '''normalize the vector p, let the sum of it become 1'''
    for i in range(50):
        p = np.dot(P.T,p)
    '''let p_new = P.Txp_old and get p_50'''
    [a,b] = np.linalg.eig(P.T)
    '''use linalg.eig to get eigenvalue and eigenvectors
    a is the eigenvalue and b is the eigenvectors'''
    p_stationary = b[:,abs(a-1)==min(abs(a-1))].real
    '''p_stationary is the eigenvectors corresponding to the eigenvalue
    closest to 1'''
    p_stationary = p_stationary/p_stationary.sum(axis=0)
    '''normalize the p_stationary'''
    if sum(abs(p-p_stationary))<(10**(-5)):
        print('The difference is less than 10^(-5)')
    else:
        print('The difference is greater than 10^(-5)')

    '''test if the difference of p and p_stationary less than 10^(-5)'''

markov_chain(5)
