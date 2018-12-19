# -- coding: utf-8 --
import numpy as np
from PSOIndividual import PSOIndividual
import random
import copy
import matplotlib.pyplot as plt


class ParticleSwarmOptimization:

    '''
    the class for Particle Swarm Optimization
    '''

    def __init__(self, sizepop, vardim, bound, MAXGEN, params):
        '''
        sizepop: population sizepop
        vardim: dimension of variables
        bound: boundaries of variables
        MAXGEN: termination condition
        params: algorithm required parameters, it is a list which is consisting of[w, c1, c2]
        '''
        self.sizepop = sizepop
        self.vardim = vardim
        self.bound = bound
        self.MAXGEN = MAXGEN
        self.params = params
        self.population = []
        self.fitness = np.zeros((self.sizepop, 1))#每一个粒子的当前适应度打分
        self.trace = np.zeros((self.MAXGEN, 2))#
        self.parameter = np.zeros((vardim))
        self.result = np.zeros((2,sizepop,vardim))
    def initialize(self):
        '''
        initialize the population of pso
        '''
        for i in xrange(0, self.sizepop):
            ind = PSOIndividual(self.vardim, self.bound)#对每一个粒子初始化
            ind.generate()#生成一个随机的
            self.population.append(ind)#将粒子加入群体中

    def evaluation(self):
        '''
        evaluation the fitness of the population
        '''
        for i in xrange(0, self.sizepop):
###
            self.population[i].calculateFitness()#计算适应度
            self.fitness[i] = self.population[i].fitness
###
            if self.population[i].fitness > self.population[i].bestFitness:#该粒子比当前最好的适应度还好
                self.population[i].bestFitness = self.population[i].fitness
                self.population[i].bestIndex = copy.deepcopy(
                    self.population[i].chrom)

    def update(self):
        '''
        update the population of pso
        '''
        for i in xrange(0, self.sizepop):
            self.population[i].velocity = self.params[0] * self.population[i].velocity + self.params[1] * np.random.random_sample(self.vardim) * (
                self.population[i].bestPosition - self.population[i].chrom) + self.params[2] * np.random.random_sample(self.vardim) * (self.best.chrom - self.population[i].chrom)
       #if self.population[i].velocity > 500:
       #     self.population[i].velocity = 500
       #elif self.population[i].velocity < -500:
       #    self.population[i].velocity = -500
            self.population[i].chrom = self.population[i].chrom + self.population[i].velocity
#            for j in xrange(0,selfvardim):
#                self.population[i].velocity[j]=min(self.population[i].velocity#[j],500)                
#                self.population[i].velocity[j]=max(self.population[i].velocity#[j],-500)                
#               self.population[i].chrom[j]=max(self.population[i].velocity#[j],0)                
            self.population[i].chrom[0]=max(self.population[i].chrom[0],0)
            self.population[i].chrom[0]=min(self.population[i].chrom[0],100)            
            self.population[i].chrom[1]=max(self.population[i].chrom[1],0)
            self.population[i].chrom[1]=min(self.population[i].chrom[1],20)            
            self.population[i].chrom[2]=max(self.population[i].chrom[2],0)
            self.population[i].chrom[2]=min(self.population[i].chrom[2],50)            
            self.population[i].chrom[3]=max(self.population[i].chrom[3],0)
            self.population[i].chrom[3]=min(self.population[i].chrom[3],100)            
     
            self.population[i].chrom[4]=max(self.population[i].chrom[4],0)
            self.population[i].chrom[4]=min(self.population[i].chrom[4],20)            
            self.population[i].chrom[5]=max(self.population[i].chrom[5],0)
            self.population[i].chrom[5]=min(self.population[i].chrom[5],50)            
            self.population[i].chrom[6]=max(self.population[i].chrom[6],0)
            self.population[i].chrom[6]=min(self.population[i].chrom[6],1)               
            self.population[i].velocity[0]=max(self.population[i].velocity[0],-20)
            self.population[i].velocity[0]=min(self.population[i].velocity[0],20)            
            self.population[i].velocity[1]=max(self.population[i].velocity[1],-4)
            self.population[i].velocity[1]=min(self.population[i].velocity[1],4)            
            self.population[i].velocity[2]=max(self.population[i].velocity[2],-10)
            self.population[i].velocity[2]=min(self.population[i].velocity[2],10)            
            self.population[i].velocity[3]=max(self.population[i].velocity[3],-20)
            self.population[i].velocity[3]=min(self.population[i].velocity[3],20)            
     
            self.population[i].velocity[4]=max(self.population[i].velocity[4],-4)
            self.population[i].velocity[4]=min(self.population[i].velocity[4],4)            
            self.population[i].velocity[5]=max(self.population[i].velocity[5],-10)
            self.population[i].velocity[5]=min(self.population[i].velocity[5],10)            
            self.population[i].velocity[6]=max(self.population[i].velocity[6],-0.2)
            self.population[i].velocity[6]=min(self.population[i].velocity[6],0.2)            


       #self.population[i].chrom = max (self.population[i].chrom,0)

    def solve(self):
        '''
        the evolution process of the pso algorithm
        '''
        self.t = 0#迭代次数
        self.initialize()
        self.evaluation()#初始化以及得到适应度
        best = np.max(self.fitness)#全局最优
        bestIndex = np.argmax(self.fitness)#全局最优的下标
        self.parameter = self.population[bestIndex].chrom 
        self.FinalIndex=bestIndex
        self.best = copy.deepcopy(self.population[bestIndex])
        self.avefitness = np.mean(self.fitness)#平均适应度
        # self.trace[self.t, 0] = (1 - self.best.fitness) / self.best.fitness#optimal function value
        self.trace[self.t, 0] = self.best.fitness#optimal function value
        # self.trace[self.t, 1] = (1 - self.avefitness) / self.avefitness#average function value is
        self.trace[self.t, 1] =  self.avefitness#average function value is
        print("Generation %d: optimal function value is: %d; average function value is %.3f" % (
            self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
        #print("current best parameter:%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f"%(self.parameter[0],self.parameter[1],self.parameter[2],self.parameter[3],self.parameter[4],self.parameter[5],self.parameter[6]))
        while self.t < self.MAXGEN - 1:
            self.t += 1
            self.update()#更新参数，根据自己的最优解和gbest
            self.params[0]=-0.005*self.t+0.9
            self.params[1]=-0.02*self.t+2.5
            self.params[2]=0.02*self.t+0.5
            self.evaluation()#评估。。。
            best = np.max(self.fitness)
            bestIndex = np.argmax(self.fitness)
            self.FinalIndex=bestIndex 
            if best > self.best.fitness:
                self.best = copy.deepcopy(self.population[bestIndex])
                self.parameter = self.population[bestIndex].chrom
                np.save('parameter.npy',self.parameter)
            self.avefitness = np.mean(self.fitness)
            # self.trace[self.t, 0] = (1 - self.best.fitness) / self.best.fitness
            # self.trace[self.t, 1] = (1 - self.avefitness) / self.avefitness
            self.trace[self.t, 0] =  self.best.fitness
            self.trace[self.t, 1] = self.avefitness
            print("Generation %d: optimal function value is: %d; current best fitness is :%d; average function value is %.3f" % (
                self.t, self.trace[self.t, 0], best, self.trace[self.t, 1]))
            #print("current best parameter:%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f"%(self.parameter[0],self.parameter[1],self.parameter[2],self.parameter[3],self.parameter[4],self.parameter[5],self.parameter[6]))
            for i in xrange(0, self.sizepop):
                self.result[1,i]=self.population[i].velocity
                self.result[0,i]=self.population[i].chrom
        print("Optimal function value is: %f; " % self.trace[self.t, 0])
        print "Optimal solution is:"
        print self.best.chrom
        self.printResult()

    def printResult(self):
        '''
        plot the result of pso algorithm
        '''
        x = np.arange(0, self.MAXGEN)
        y1 = self.trace[:, 0]
        y2 = self.trace[:, 1]
        plt.plot(x, y1, 'r', label='optimal value')
        plt.plot(x, y2, 'g', label='average value')
        plt.xlabel("Iteration")
        plt.ylabel("function value")
        plt.title("Particle Swarm Optimization algorithm for function optimization")
        plt.legend()
        plt.show()
