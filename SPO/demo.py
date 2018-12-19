# -- coding: utf-8 --
import numpy as np
import CartPoleControl 
import matplotlib.pyplot as plt

chrom = np.load('parameter.npy')
fitness = CartPoleControl.CratpoleControl(chrom,100,10000,1)
average_fitness = np.mean(fitness)
print "average function value is %.3f" %average_fitness
x = np.arange(0,len(fitness))
ave = np.ones(len(fitness))
plt.plot(x, fitness,'g', label='single value')
plt.plot(x, ave*average_fitness, 'r', label='average value')
plt.xlabel("time")
plt.ylabel("function value")
plt.title("PSO algorithm for function optimization")
plt.legend()
plt.show()
