#!/usr/bin/python
#author:Jiecheng Song SBID:111783762

import matplotlib.pyplot as plt
import numpy as np

def mandelbrot_1(N_max,some_threshold,m,n):
    '''define a function called mandelbrot_1 with parameter N_max,
    some_threshold, m, n'''
    x,y = np.mgrid[-2:1:complex(0,m),-1.5:1.5:complex(0,n)]
    '''use np.mgrid funtion to construct a grid (x,y) in range
    [-2,1]x[-1.5,1.5], x and y are both mxn matrix'''
    c = x + 1j*y
    '''make a mxn complex matrix c each entrance is c_ij = x(i)+y(j)i'''
    z = np.zeros([m,n])
    '''make a mxn zero matrix z'''
    mask = np.zeros([m,n])
    '''make a mxn zero matrix mask, so the boolean mask are all false now'''
    for j in range(N_max):
        z = z**2 + c
        mask[abs(z)<some_threshold] = j
    '''use for loop to do the interation z = z**2+c 50 times, let boolean mask
    z_ij become j if at the jth times, abs(z_ij) over 50'''
    print(mask)
    '''print boolean maske matrix'''
    plt.figure(2)
    plt.imshow(mask.T,extent=[-2,1,-1.5,1.5])
    plt.gray()
    plt.savefig('mandelbrot_1.png')
    '''save the figure'''
    plt.show()
    '''plot the boolean mask with the bigger value more white'''


mandelbrot_1(50,50,1000,1000)
'''execute the function'''
