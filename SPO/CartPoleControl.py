import gym
import numpy as np
import math
env = gym.make('CartPole-v1')
def CratpoleControl(weigth,episode, step_limit, isprint):
	weigth1=weigth[0:80];
	weigth1=np.reshape(weigth1,(4,20))
	weigth2=weigth[80:120];
	weigth2=np.reshape(weigth2,(20,2))
	tt = []
	for i_episode in range(episode):
		observation = env.reset()
		for t in range(step_limit):
		    env.render()
		    action = env.action_space.sample() 
		    observation, reward, done, info = env.step(action)
		    hidden_output=np.dot(observation,weigth1)  
		    hidden_output=[1/(1+math.exp(-j)) for j in hidden_output]
		    output=np.dot(hidden_output,weigth2)
		    if output[0]>output[1]:
		        action=1;
		    else:
		        action=0;
		    if done:
		    	if isprint:
		        	print("Episode finished after {} timesteps".format(t+1))
		        tt.append(t)
		        break
		if isprint and t==step_limit:
			print("Episode finished after {} timesteps".format(t+1))
			tt.append(t)
	return tt
