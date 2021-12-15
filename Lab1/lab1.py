import scipy.stats as st
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import statistics as s

import operator as op
from functools import reduce

# DREW GOLDMAN dag5wd
# QUESTION 1


def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom


# normal RV with mean 0 and variance 1 (standard deviation 1)
mynorm = st.norm(0, 1)
r = np.linspace(-10, 10, 1000)
p = .5

pvalue_a = st.binom.cdf(2, 5, .5)
print("The p-value for part a is: " + str(pvalue_a))
p_a = (ncr(5, 3) * p**3 * (1-p)**2) + (ncr(5, 4) * p **
                                       4 * (1-p)**1) + (ncr(5, 5) * p**5 * (1-p)**0)
print("The probability of flipping at least three heads out of five coin flips is: " + str(p_a))

pvalue_b = st.binom.cdf(4, 10, .5)
print("The p-value for part b is: " + str(pvalue_b))

p_b = (ncr(10, 6) * p**6 * (1-p)**4) + (ncr(10, 7) * p ** 7 * (1-p)**3) + (ncr(10, 8)
                                                                           * p**8 * (1-p)**2) + (ncr(10, 9) * p**9 * (1-p)**1) + (ncr(10, 10) * p**10 * (1-p)**0)
print("The probability of flipping at least six heads out of ten coin flips is: " + str(p_b))


partC = []
for i in range(100):
    if(i < 60):
        partC.append(1)
    else:
        partC.append(0)
p_hat_c = sum(partC) / len(partC)
print(p_hat_c)
z_c = (p_hat_c - p) / (np.sqrt(p*(1-p)/len(partC)))
print("The z-value for part c is: " + str(z_c))

p_c = mynorm.sf(z_c)
print("The p-value for part c is: " + str(p_c))

partD = []
for i in range(100):
    if(i < 40):
        partD.append(1)
    else:
        partD.append(0)
p_hat_d = sum(partD) / len(partD)
print(p_hat_d)
z_d = (p_hat_d - p) / (np.sqrt(p*(1-p)/len(partD)))
print("The z-value for part d is: " + str(z_d))

p_d = mynorm.sf(z_d)
print("The p-value for part d is: " + str(p_d))


# QUESTION 2
"""
Naive Bayes Approach
p(Y = y|X = x) = p(X = x|Y = y)p(y) / p(x) the p(x|y) will be a binomial random variable with y as n and p as 1/2 (you're flipping y coins)
p(x | y) = (y choose x)(1/2)^x*(1/2)^(y-x) = (y choose x) * 2^(-y)
p(y) = 1/6
to get probability of x use total law summation over all possible values of p(x|y)*p(y) ==>
summation of y' = x to 6 (because y', the number of coin flips, cannot be less than the number of heads) of (y' choose x) * (1/2)^(y') * (1/6)
maximize that ((y choose x) * (1/2)^y) / ((y' choose x) * (1/2)^y')
for each x compute the probability for each y and see which one gives the highest value (for all 36 combinations)
I'm expecting to get y_hat = 2x (expected number of times you flip the coin is twice the amount of heads approximately)
"""

lst = [1, 2, 3, 4, 5, 6]
lst2 = [0, 1, 2, 3, 4, 5, 6]
theX, theY = 0, 0
for x in lst2:
    maximum = -1
    for y in lst:
        if(y >= x):
            comb = ncr(y, x)
            p_x_given_y = (.5**y)
            p_x = sum(ncr(y_pr, x)*(1/2)**y_pr for y_pr in range(x, 7))
            pyx = (comb * p_x_given_y) / p_x
            if(pyx >= maximum):
                maximum = pyx
                theX = x
                theY = y
            print("for x = " + str(x) + ", y = " + str(y) + ": " + str(pyx))
    print("Maximum probability: " + str(maximum) +
          " for x = " + str(theX) + ", y = " + str(theY))


# QUESTION 4
E_Y = 7/2
E_Ysquared = 0
for i in lst:
    E_Ysquared += (1/6)*(i**2)
Var_Y = E_Ysquared - (E_Y)**2  # Var_Y = 35/12
sigma_Y = np.sqrt(Var_Y)  # sigma_Y = 1.7078
E_X = 0
for x in lst2:
    p_x = (1/6)*sum(ncr(y_pr, x)*(1/2)**y_pr for y_pr in range(x, 7))
    E_X += p_x*x
E_Xsquared = 0
for x in lst2:
    if y >= x:
        p_x = (1/6)*sum(ncr(y_pr, x)*(1/2)**y_pr for y_pr in range(x, 7))
        E_Xsquared += p_x*(x**2)
Var_X = E_Xsquared - (E_X)**2  # Var[X] = 1.6042 = 4.666 - 1.75^2
sigma_X = np.sqrt(Var_X)  # sigma_X = 1.2666
#print("E[X^2] = " + str(E_Xsquared))
#E[XY] = x1y1*p(x1,y1) + x1y2*p(x1,y2) + ... + xnyn*p(xn,yn)
# p(x,y) = p(x|y)*p(y)
E_XY = 0
for x in lst2:
    maximum = -1
    for y in lst:
        if(y >= x):
            comb = ncr(y, x)
            p_x_given_y = (.5**y) * comb
            p_x_and_y = p_x_given_y * (1/6)
            #print("for x = " + str(x) + " and y = " + str(y) + " p = " + str(p_x_and_y))
            E_XY += (p_x_and_y * x * y)
            # print(E_XY)
Cov_XY = E_XY - (E_X*E_Y)  # 1.4583 = 7.5833 - 1.75*(7/2)
rho = Cov_XY / (sigma_X*sigma_Y)  # .6742
optimalL = Var_Y*(1-rho**2)
print("Optimal loss: " + str(optimalL))
for x in lst2:
    f_X = (rho*sigma_Y/sigma_X)*(x - E_X) + E_Y
    print("for x = " + str(x) + ", f*(X) = " + str(f_X))

# print((rho*sigma_Y/sigma_X))
# print(E_Y-E_X*(rho*sigma_Y/sigma_X))
# print(E_Y)
