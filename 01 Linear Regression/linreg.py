##Coursera -ML
##Exercise -1 
##python version 3.6
##directory structure--> code= '/coursera/scripts' ; data = '/coursera/data'


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import axes3d
#%matplotlib inline  

import seaborn as sn

##Linear regression in one variable

data = np.loadtxt('../data/ex1data1.txt', delimiter=',')
X = np.c_[np.ones(data.shape[0]),data[:,0]]
y = np.c_[data[:,1]]

## Plotting the scatter plot

plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1)
plt.xlim(4,24)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s');


##defining function for gradient descent

def CostFunc(X, y, beta=[[0],[0]]):
    m = y.size
    J = 0
    h = X.dot(beta)     ##hypothesis is h(beta) = X(t)*beta
    J = 1/(2*m)*np.sum(np.square(h-y))    ##sum of squares of residuals averaged across observations
    return(J)

##testing Costfunc
CostFunc(X,y)


def GradientDescent(X, y, beta=[[0],[0]], alpha=0.01, num_iters=1000):
    """defining vanilla gardient descent to update the beta with rate of change of J"""
    m = y.size
    J_so_far = np.zeros(num_iters)
    for i in np.arange(num_iters):
        h = X.dot(beta)
        beta = beta - alpha*(1/m)*(X.T.dot(h-y))
        J_so_far[i] = CostFunc(X, y, beta)
    return(beta, J_so_far)



## evaluating betas with every iteration 
## this will give an idea of convergence with every iteration
beta_obtained , J_val = GradientDescent(X, y)
print('betas: ',beta_obtained.ravel())

plt.plot(J_val)
plt.ylabel('Cost function values')
plt.xlabel('Iterations');


## Running comparisons with sklearn

xx = np.arange(5,23)
yy = beta_obtained[0]+beta_obtained[1]*xx

# Plot gradient descent
plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1)
plt.plot(xx,yy, label='Linear regression (Gradient descent)')

# Compare with Scikit-learn Linear regression 
reg_mod = LinearRegression()
reg_mod.fit(X[:,1].reshape(-1,1), y.ravel())
a0 = reg_mod.intercept_
my_betas = reg_mod.coef_

plt.plot(xx, a0+my_betas*xx, label='Linear regression (Scikit-learn)')

plt.xlabel('Population of City (10,000s)')
plt.ylabel('Profit ($10,000s)')
plt.legend(loc=4);

print(beta_obtained.shape)
##making predictions
##a) a population = 35k, and b= 70k
print("profit for city a ",beta_obtained.T.dot([1, 3.5])*10000)
print("profit for city b ",beta_obtained.T.dot([1, 7])*10000)

##making contour plot for intuitive visualization

# Create grid coordinates for plotting
B0 = np.linspace(-10, 10, 50)
B1 = np.linspace(-1, 4, 50)
xx2, yy2 = np.meshgrid(B0, B1, indexing='xy')
Z = np.zeros((B0.size,B1.size))

# Calculate Z-values (Cost) based on grid of coefficients
for (i,j),v in np.ndenumerate(Z):
    Z[i,j] = CostFunc(X,y, beta=[[xx2[i,j]], [yy2[i,j]]])

fig = plt.figure(figsize=(15,6))
ax1 = fig.add_subplot(121)


#contour plot
CS = ax1.contour(xx2, yy2, Z, np.logspace(-2, 3, 20), cmap=plt.cm.jet)
ax1.scatter(beta_obtained[0],beta_obtained[1], c='r')
ax1.set_xlabel(r'$\theta_0$', fontsize=17)
ax1.set_ylabel(r'$\theta_1$', fontsize=17)




