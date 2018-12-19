# -- coding: utf-8 --
import numpy as np
import CartPoleControl 
import matplotlib.pyplot as plt

chrom = np.load('parameter.npy')
fitness = CartPoleControl.CratpoleControl(chrom[0],chrom[1],chrom[2],chrom[3],chrom[4],chrom[5],chrom[6],100,10000,1)
average_fitness = np.mean(fitness)
print "average function value is %.3f" %average_fitness
x = np.arange(0,len(fitness))
ave = np.ones(len(fitness))
plt.plot(x, fitness,'g', label='single value')
plt.plot(x, ave*average_fitness, 'r', label='average value')
plt.xlabel("time")
plt.ylabel("function value")
plt.title("PID algorithm for function optimization")
plt.legend()
plt.show()

'''
fitness1 = np.ones(100)*500
fitness2 = np.ones(100)*500
x = np.arange(0,100)
plt.plot(x, fitness1, 'g', label='single value')
plt.plot(x, fitness2, 'r', label='average value')
plt.xlabel("time")
plt.ylabel("function value")
plt.title("PID algorithm for function optimization")
plt.legend()
plt.show()
'''
