#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 16:16:22 2020

@author: ansonpoon
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from scipy.interpolate import CubicSpline
from scipy import interpolate
from computational_functions import *

#%%
#Question 1a
num=0.25
lower_difference = find_lower_error(num)
upper_difference = find_upper_error(num)
fractional_range = ((upper_difference + lower_difference)/2)/num

print('\n')
print(f'The error associated with the lower difference of 0.25 is {lower_difference}.')
print(f'The error associated with the upper difference of 0.25 is {upper_difference}.')
print('\n')
print(f'The fractional rounding range of 0.25 is {fractional_range}.')


#%%
#Question 1b

upper_bound = 0.25 + find_upper_error(0.25)
lower_bound = 0.25 - find_lower_error(0.25) 

print('\n')
print(f'The nearest values of 0.25 are {(upper_bound, lower_bound)}.')

print('\n')
print(f'The nearest values of the upperbound are {nearest_values(upper_bound)}.')
print(f'The nearest values of the lowerbound are {nearest_values(lower_bound)}.')

fractional_range_upper_bound = ((find_upper_error(upper_bound) + find_lower_error(upper_bound)))/upper_bound
fractional_range_lower_bound = ((find_upper_error(lower_bound) + find_lower_error(lower_bound)))/lower_bound

print('\n')
print(f'The fractional rounding range of the upper bound is {fractional_range_upper_bound}.')
print(f'The fractional rounding range of the lower bound is {fractional_range_lower_bound}.')

#%%
#Validation for nearest values

print('\nValidation using numpy built-in function')
print('\n')
print(f'The nearest values of 0.25 are {(np.nextafter(0.25, 0.26),np.nextafter(0.25, 0.24))}.')

print('\n')
print(f'The nearest values of the upperbound are {(np.nextafter(upper_bound, 0.26), np.nextafter(upper_bound, 0.25))}.')
print(f'The nearest values of the lowerbound are {(np.nextafter(lower_bound, 0.25), np.nextafter(lower_bound, 0.23))}.')




#%%
#Question 2b
A = np.array([[3,1,0,0,0],
              [3,9,4,0,0],
              [0,8,20,10,0],
              [0,0,-22,31,-25],
              [0,0,0,-35,61]])
L, U, M = crouts(A)

determinant = 1

for i in range(len(U)): 
    determinant *=  U[i,i]
print('\n')
print('The lower matrix is:')
print(L)
print('\n')
print('The upper matrix is:')
print(U)
print('\n')
print(f'The determinant is {determinant}.')

#%%
#Validation of the determinant

print('\nValidation of the determinant of A')
print(f'The determinant calculated from the numpy built-in fucntion is {np.linalg.det(A)}')


#%%
#Question 2d
A = np.array([[3,1,0,0,0],
              [3,9,4,0,0],
              [0,8,20,10,0],
              [0,0,-22,31,-25],
              [0,0,0,-35,61]])
b = np.array([2,5,-4,8,9])

L, U, M = crouts(A)

print('\n')
print(f'The solution is {solve_lu(L,U,b)}.')
#%%
#Validation of lu_solve
#This shows A can be obtained multiply L with U.
print(f'L*U = \n{np.dot(L,U)} \n= A')

#%%
#Question 2e

A = np.array([[3,1,0,0,0],
              [3,9,4,0,0],
              [0,8,20,10,0],
              [0,0,-22,31,-25],
              [0,0,0,-35,61]])

A_inverse = np.zeros(A.shape) #This initialises the inverse matrix to be a zero matrix.
identity = np.identity(len(A)) 
    
L, U, M = crouts(A)
#This for loop calculates the values in the inverse matrix column by column using the solve_lu function from above.
for i in range(len(A)):
    A_inverse[:,i] = solve_lu(L,U,identity[i,:])

print('\n')
print('The inverse of A is:')
print(A_inverse)    

#%%
#Validation of the inverse

print('\nValidation of the inverse of A')
print(f'The inverse calculated from the numpy built-in fucntion is \n{np.linalg.inv(A)}.')


#%%
#Question 3c

x_data = [-0.75,-.5,-0.35,-0.1,0.05,0.1,0.23,0.29,0.48,0.6,0.92,1.05,1.5]
y_data = [0.10, 0.30, 0.47, 0.66, 0.60, 0.54, 0.30, 0.15, -0.32, -0.54, -0.60, -0.47, -0.08]

x = np.linspace(x_data[0], x_data[-1], 1000, endpoint = False)
y_cubic = [cubic_spline(i, x_data, y_data) for i in x]
y_lagrange = [lagrange_pol(i, x_data, y_data) for i in x]





plt.plot(x, y_cubic, 'b', label = 'cubic spline')
plt.plot(x, y_lagrange, 'r', label = 'Lagrange polynomials')
plt.scatter(x_data, y_data, c = 'green', marker = 'x', label = 'Given data points')
plt.xlabel('x', fontsize = 25)
plt.ylabel('y',  fontsize = 25)
plt.title('Interpolation techniques',  fontsize = 25)
plt.legend(loc=(0.8,1), fontsize = 10)
plt.show()

#%%
#Validation of lagrange_pol
lag = lagrange(x_data, y_data)
scipy_lagrange = [lag(i) for i in x]
plt.plot(x, scipy_lagrange, label = 'Scipy lagrange polynomials')
plt.plot(x, y_lagrange, 'r', label = 'lagrange_pol function')
plt.title('Validation of the Lagrange polynomial function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

#%%
#Validation of cuboic_spline
cub= CubicSpline(x_data, y_data)
scipy_cubic = [cub(i) for i in x]
plt.plot(x, scipy_cubic, label = 'Scipy cubic spline')
plt.plot(x, y_cubic, 'r', label = 'cubic_spline function')
plt.title('Validation of the cubic spline function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()



#%%
#Question 4
def g(t):
    constant = (2*np.pi)**-0.5
    exp = np.exp(-(t**2)/4)
    
    return constant * exp

def h(t):
    if t<=7 and t>=5:
        return 4
    else:
        return 0

n = 13
t = np.linspace(-20,20, num=2**n) #



zeros = [0 for x in range(2**n)]

g_list = [g(x) for x in t]
h_list = [h(x) for x in t]
g_zero_pad = [g(x) for x in t] + [0 for x in range(2**n)]#adds zeros for zero padding
h_zero_pad = [h(x) for x in t] + [0 for x in range(2**n)]

g = np.array(g_zero_pad)
h = np.array(h_zero_pad)

g_fourier = np.fft.fft(g)
h_fourier = np.fft.fft(h)


product = g_fourier*h_fourier

step = t[1] - t[0]

inverse = np.fft.ifft(product)
convolution = inverse[int(2**n/2):-int(2**n/2)]*step #This removes the zeros data points
plt.plot(t, convolution, label='Convoluted function (g*h)')
plt.plot(t, g_list, label='g(t)')
plt.plot(t, h_list, label='h(t)')
plt.title('Convolution of the signal function')
plt.xlabel('t')
plt.ylabel('f(t)')
plt.legend()
plt.show()


#%%
#Question 5c

V_0 = 10

def V_in_step(t):
    if t<=0:
        return V_0 
    else:
        return 0
    
def f(t, V_out):
    return (V_in_step(t) - V_out)

t_ab, V_out_ab = ode_methods(10, 1000, f, 0, 10, Adams_Bashforth = True, Runge_Kutta = False)
t_rk, V_out_rk= ode_methods(10, 1000, f, 0, 10, Adams_Bashforth = False, Runge_Kutta = True)

def func_an(t):
    return V_0*np.exp(-t)

y_an = [func_an(i) for i in t_ab]


plt.plot(t_ab,V_out_ab, label = 'Adams-Bashforth method', c= 'r')
plt.plot(t_rk, V_out_rk, label = 'Runge-Kutta method', c = 'b')
plt.plot(t_ab, y_an, label = 'Analytic solution')
plt.title('Approximate and Analytic Solutions for V_out')
plt.ylabel('V_out(t) (V)')
plt.xlabel("Scaled time t'")


plt.legend()
plt.show()


#%%
#Question 5d
def func(x,y): 
    return -y

t_original, V_out1 = ode_methods(10, 1000, func, 0, 10, Adams_Bashforth = False, Runge_Kutta = True)
t_double, V_out2 = ode_methods(10, 500, func, 0, 10, Adams_Bashforth = False, Runge_Kutta = True)
t_half, V_out3 = ode_methods(10, 2000, func, 0, 10, Adams_Bashforth = False, Runge_Kutta = True)

def analytical_solution(t):
    return 10*np.exp(-t)
V_out_analytical1 = np.array([analytical_solution(i) for i in t_original])
V_out_analytical2 = np.array([analytical_solution(i) for i in t_double])
V_out_analytical3 = np.array([analytical_solution(i) for i in t_half])

diff1 = V_out1 - V_out_analytical1
diff2 = V_out2 - V_out_analytical2
diff3 = V_out3 - V_out_analytical3

plt.plot(t_original, diff1, label = 'stepsize = 0.01')
plt.plot(t_double, diff2, label = 'stepsize = 0.02')
plt.plot(t_half, diff3, label = 'stepsize = 0.005')
plt.xlabel("Scaled time t'")
plt.ylabel('Error')
plt.title('Errors compared to the analytic solution')
plt.legend()
plt.show()

#%%
#Question 5d continued
#Here the ratios of the erros between different step sizes are calculated and plotted
double_to_original_ratio = []
for i in range(len(t_double)):
    double_to_original_ratio.append(diff2[i]/ diff1[2*i])

original_to_half_ratio = []
for i in range(len(t_original)):
    original_to_half_ratio.append(diff1[i]/diff3[2*i])

plt.plot(t_double, double_to_original_ratio, label = 'error of the double step size : error of original step size')
plt.plot(t_original, original_to_half_ratio, label = 'error of the original step size : error of half step size')
plt.ylim((0,18))
plt.xlabel("Scaled time t'")
plt.ylabel('Ratios of the error')
plt.title('Ratio of error')
plt.legend()
plt.show()
    

#%%
#Question 5e

RC = 1
V_0 = 10
T = RC/2
#T_2 = RC/2
def V_in_square(t):
    while t>T:
        t -= T
    if t<0:
        return V_0
    elif t>=0 and t< T/2:
        return 0
    elif t>= T/2 and t<T:
        return V_0
    
def f(t, V_out):
    return (V_in_square(t) - V_out)


    
t_2, V_2 = ode_methods(10, 1000, f, x_0=0, y_0= V_0, Adams_Bashforth = False, Runge_Kutta = True)
T = RC*2
t_1, V_1 = ode_methods(10, 1000, f, x_0=0, y_0= V_0, Adams_Bashforth = False, Runge_Kutta = True)
plt.plot(t_2,V_2, label = 'T = RC/2')
plt.plot(t_1,V_1,label= 'T = 2RC' )
plt.xlabel("Scaled time t'")
plt.ylabel('V_out(t) (V)')
plt.title('Solution of V_out with a square wave V_in')
plt.legend()
plt.show()

















