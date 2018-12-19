# -- coding: utf-8 --
import CartPoleControl 
import numpy as np
import PSO
if __name__ == "__main__":
 
     bound = np.array(([[0], [100]], [[0], [20]], [[0], [50]], [[0], [100]], [[0], [20]], [[0], [50]], [[0], [1]]))
     pso = PSO.ParticleSwarmOptimization(20, 7, bound, 100, [0.7298, 1.4962, 1.4962])
     pso.solve()
     bestsolve=pso.parameter
     print bestsolvem#是输出的最优解
     Finalfitness = CartPoleControl.CratpoleControl(bestsolve[0],bestsolve[1],bestsolve[2],bestsolve[3],bestsolve[4],bestsolve[5],bestsolve[6])
#60 个 粒子（解），每个解有25个维度，每个维度的范围在[-600,600],迭代1000结束，
#最后是更新的参数[w, c1, c2]：v = w*v +c1*(pbest-pop)+c2*(gbest-pop)w表示粒子过去的速度所占的权重，
#c1,c2表示上面所说的两个参数，gbest表示粒子过去所达到的最好的适应度值对应的位置，
#gbest表示粒子群中的首领过去达到过的最好的适应度值对应的位置,pop表示粒子当前的位置
