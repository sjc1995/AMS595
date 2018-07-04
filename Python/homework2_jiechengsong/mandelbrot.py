#!/usr/bin/python
#author:Jiecheng Song SBID:111783762

import matplotlib.pyplot as plt
import numpy as np

def mandelbrot(N_max,some_threshold,m,n):
    '''define a function called mandelbrot with parameter N_max,
    some_threshold, m, n'''
    x,y = np.mgrid[-2:1:complex(0,m),-1.5:1.5:complex(0,n)]
    '''use np.mgrid funtion to construct a grid (x,y) in range
    [-2,1]x[-1.5,1.5], x and y are both mxn matrix'''
    c = x + 1j*y
    '''make a mxn complex matrix c each entrance is c_ij = x(i)+y(j)i'''
    z = np.zeros([m,n])
    '''make a mxn zero matrix z'''
    for j in range(N_max):
        z = z**2 + c
    '''use for loop to do the interation z = z**2+c 50 times'''
    mask = np.zeros([m,n])
    '''make a mxn zero matrix mask, so the boolean mask are all false now'''
    mask[abs(z)<some_threshold] = 1
    '''if the abs(z_ij)<some_threshold, make z_ij = 1 so the boolean value
    becoming true'''
    print(mask)
    '''print the boolean mask'''
    plt.figure(1)
    plt.imshow(mask.T,extent=[-2,1,-1.5,1.5])
    plt.gray()
    plt.savefig('mandelbrot.png')
    '''save the figure we plot'''
    plt.show()
    '''plot the boolean mask with true is white and false is black'''

mandelbrot(50,50,1000,1000)
'''execute the function'''

'''after finish the function, I found the figure I plot is different with the figure
professor gave us, I do not get the line on the left, so I read some information and
ask some classmates why the fugures are different, then I rewrite another funtion.
In the other function, we can get the line.'''
