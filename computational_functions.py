#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 16:14:49 2020

@author: ansonpoon
"""

'''Import Numpy'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from scipy.interpolate import CubicSpline
from scipy import interpolate


#%%
#Question 1a
def find_upper_error(num=1):
    '''
    This function takes in a number and returns the associated computational error from the upper limit.
    
    Args:
        num: An int or float
    '''
    
    error = num 
    run = True
    lst = []
    while run: #This while loop adds a new error value to the initial value.
        number = num
        error /= 2 #The error is divided because of binary system.
        number +=error
        lst.append(error)
        if number == num:
            run = False #The while loop stops, when the new value is the same as the previous one.
    return lst[-2]
            

def find_lower_error(num=1):
    '''
    This function takes in a number and returns the associated computational error from the lower limit.
    
    Args:
        num: An int or float
    '''
    error = 1.0
    run = True
    lst = []
    while run:
        number = num
        error /= 2
        number -=error
        lst.append(error)
        if number == num:
            run = False
    return lst[-2]
            


#%%
def nearest_values(num):
    '''
    This function returns the upper bound and lower bound of the input number.
    
    args:
        num: An int or float
    '''
    return num + find_upper_error(num), num - find_lower_error(num)




#%%
#Question 2a
def crouts(A):
    '''
    This function factorises the input matrix A into a lower diagonal matrix L
    and an upper diagonal matirx U, using Crout's alogorithm.
    
    Args:
        A: A numpy array representing any square matrix.
        
    Returns:
        A tuple containg the lower diagonal matrix L, 
        an upper diagonal matirx U and a matrix M that contains all the
        elements of U and all the non-diagonal elements of L.
    '''
    
    dimension = A.shape 
    m = len(A)
    #The lower diagonal and upper diagonal matrices are initialised
    L = np.identity(m) 
    U = np.zeros(dimension)
    for j in range(m): #This computes L and U column by column
        for i in range(m): #This computes U row by row
            if i <=j:
                U[i,j] = A[i,j]
                for k in range(i):
                    U[i,j] -= L[i,k]*U[k,j]
                    
            else:
                L[i,j] = A[i,j]/U[j,j]#This computes L row by row
                for k in range(j): 
                    L[i,j] -= L[i,k]*U[k,j]/U[j,j]
    #The matrix M contains all the the elements of U and the non-diagonal elements of L
    #An identity matrix is subtracted from the sum of L and U to remove the diagonal elements of L.
    M = L + U - np.identity(m) 
    return L, U, M
    
    



#%%
#Question 2c
def solve_lu(L,U,b):
    '''
    This function solves x in the matrix equation L*U*x=b, where L is an lower diagonal matrix,
    U is an upper diagonal matrix and b is a one dimensional vector.
    
    Args:
        L: An numpy array representing the lower diagonal matrix.
        U: An numpy array representing the upper diagonal matrix.
        b: An numpy array representing the vector on the right hand side of the equation.
    '''
    
    m = len(U)
    y = np.zeros(m)
    y[0] = b[0]
    #This solves for U*x using forward substitution.
    for i in range(1,m):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i,j]*y[j]
            
    x = np.zeros(m)
    x[m-1] = y[m-1]/U[m-1,m-1]
    #This solves for x using forward substitution.
    for i in reversed(range(m-1)):
        x[i] = y[i]/U[i,i]
        for j in range(i+1,m):
            x[i] -= U[i,j]*x[j]/U[i,i]
            
    return x



#%%
#Question 3a
def lagrange_pol(x_input, x_data, y_data):
    '''
    This function interpolates a set of x and y data, using Lagrange polynomials.
    
    Args:
     x_input: Int or float, where the x value where it is interpolated.
        x_data: Numpy array, representing the set of x values from the data set.
        y_data: Numpy array, representing the set of y values from the data set.
    
    Returns:
        The interpolated value at x_input.
    '''
    
    n = len(x_data)
    
    polynomial = 0
    for i in range(n):
        product = 1
        for j in range(n):
            if i != j:
                product *= (x_input - x_data[j])/(x_data[i] - x_data[j])
    
        polynomial += product*y_data[i] 
        
    return polynomial



#%%
#Question 3b
def cubic_spline(x_input, x_data, y_data):
    '''
    This function interpolates a set of x and y data, using cubic spline method.
    
    Args:
        x_input: Int or float, where the x value where it is interpolated.
        x_data: Numpy array, representing the set of x values from the data set.
        y_data: Numpy array, representing the set of y values from the data set.
    
    Returns:
        The interpolated value at x_input.
    '''
    
    n = len(x_data)


    

    M = np.zeros((n,n)) #This sets creates a matrix which can the solve for the second derivatives.
    for i in range(1, len(M)-1):
        M[i,i-1] = (x_data[i] -x_data[i-1])/6
        M[i,i] = (x_data[i+1]-x_data[i-1])/3
        M[i,i+1] = (x_data[i+1] - x_data[i])/6
    M[0,0] = 1 
    M[n-1, n-1] = 1#These add rows to the matrix to make it a square matrix so LU decomposition can be used.
    
    b = np.zeros(n) #This creates the vector that LU decomposition algorithm is solving against.
    for i in range(1,n-1):
        b[i] = (y_data[i+1] - y_data[i])/(x_data[i+1]-x_data[i])-(y_data[i] - y_data[i-1])/(x_data[i]-x_data[i-1])
    
    L, U, W = crouts(M)
    f2 = solve_lu(L,U, b)

    for i in range(len(x_data)):
        if x_input >= x_data[i] and x_input<=x_data[i+1]: #This is to ensure which range the value falls under.
            A = (x_data[i+1] - x_input)/(x_data[i+1]-x_data[i])
            B = 1 - A
            h = x_data[i+1] - x_data[i]
            C = (A**3 - A)*(h**2)/6
            D = (B**3 - B)*(h**2)/6
            answer = A*y_data[i] + B*y_data[i+1] + C*f2[i] + D*f2[i+1] #This expresses f(x) in terms of its second derivatives.
            
    return answer

#%%
#Question 5b


def ode_methods(endpoint, num_steps, f, x_0, y_0, Adams_Bashforth = True, Runge_Kutta=False):
    '''
    This function returns the approximate solution of any first order differential equation, 
    using Adams_Bashforth or Runge_Kutta method.
    
    Args:
        endpoint: An int or float, at where your x values end.
        num_steps: An int, representing how many steps to take.
        f: A function, representing the first derivative.
        x_0: An int or float, representing the initial x value.
        y_0: An int or float, representing the initial y value.
        Adams_Bashforth: A boolean value. If true, the Adams_Bashforth method would be used.
        Runge_Kutta: A boolean value. If true, the Runge_Kutta method would be used.
        
    Returns:
        A set of x values and a set of y values as two numpy arrays.
    
    '''
    
    num_terms = num_steps +1
    
    stepsize = (endpoint-x_0)/num_steps
    
    #These initialise the x and y arrays.
    x = np.zeros(num_terms)
    x[0] = x_0
    y = np.zeros(num_terms)
    y[0] = y_0
    
    for i in range(num_steps):
        x[i+1] = x[i] + stepsize


    if Adams_Bashforth:
        y[1] = y[0] + f(x[0], y[0])*stepsize #This finds the second y value using Euler's method.
        for i in range(1, 3): #This finds the other initial values by the leapfrog method.
            y[i+1] = y[i-1] + f(x[i], y[i])*stepsize*2
        
        for i in range(3, num_terms-1): #This creates the iterative equation for the Adams-Bashforth method.
            f_a = f(x[i], y[i])
            f_b = f(x[i-1], y[i-1])
            f_c = f(x[i-2], y[i-2])
            f_d = f(x[i-3], y[i-3])
            y[i+1] = y[i] + (stepsize/24)*(55*f_a -59*f_b + 37*f_c - 9*f_d)
            
    elif Runge_Kutta:
        for i in range(0, num_terms-1):
            fa = f(x[i],y[i])
            fb = f(x[i] + stepsize/2, y[i]+ stepsize * fa/2)
            fc = f(x[i] +stepsize/2, y[i] + stepsize*fb/2)
            fd = f(x[i] + stepsize, y[i] + stepsize* fc)
            
            y[i+1] = y[i] + (stepsize/6)*(fa +2*fb + 2*fc + fd)

    return x,y


