# -- coding: utf-8 --
import numpy as np
import copy
import CartPoleControl 

class PSOIndividual:

    '''
    individual of PSO
    '''

    def __init__(self, vardim, bound):
        '''
        vardim: dimension of variables
        bound: boundaries of variables
        '''
        self.vardim = vardim
        self.bound = bound
        self.fitness = 0.

    def generate(self):
        '''
        generate a rondom chromsome
        '''
        len = self.vardim
        rnd = np.random.random(size=len)
        self.chrom = np.zeros(len)#位置
        self.velocity = np.random.random(size=len)#速度
        for i in xrange(0, len):
            self.chrom[i] = self.bound[i, 0] + \
                (self.bound[i, 1] - self.bound[i, 0]) * rnd[i]#使得位置（解）映射到区间内部
        self.bestPosition = np.zeros(len)
        self.bestFitness = 0.
###
    def calculateFitness(self):
        '''
        calculate the fitness of the chromsome
        '''
        # self.fitness = ObjFunction.GrieFunc(
            # self.vardim, self.chrom, self.bound)
        fit = CartPoleControl.CratpoleControl(self.chrom[0],self.chrom[1],self.chrom[2],self.chrom[3],self.chrom[4],self.chrom[5],self.chrom[6],1,400,0)
        self.fitness = np.mean(fit)
###
