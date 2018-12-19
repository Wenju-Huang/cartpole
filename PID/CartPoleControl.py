import gym
env = gym.make('CartPole-v1')
def CratpoleControl(p_x, i_x, d_x, p_theta, i_theta, d_theta, lamb, episode, step_limit, isprint):
	tt = []
	for i_episode in range(episode):
		observation = env.reset()
		v_x = 0
		x_t = x_t1 = x_t2 = 0
		v_theta = 0
		theta_t = theta_t1 = theta_t2 = 0
		for t in range(step_limit):
		    env.render()
		    v_x = v_x + p_x*(x_t - x_t1) + d_x*(x_t - 2*x_t1 +x_t2) + i_x*x_t
		    v_theta = v_theta + p_theta*(theta_t - theta_t1) + d_theta*(theta_t - 2*theta_t1 +theta_t2) + i_theta*theta_t
		    v = lamb*v_x + (1-lamb)*v_theta;
		    if v > 0:
		    	action = 1
		    else:
		    	action = 0
		    #state = (x,x_dot,theta,theta_dot) 
		    observation, reward, done, info = env.step(action)  
		    x_t2 = x_t1
		    x_t1 = x_t
		    x_t = observation[0]
		    theta_t2 = theta_t1
		    theta_t1 = theta_t
		    theta_t = x_t = observation[2]
		    if done:
		    	if isprint:
		        	print("Episode finished after {} timesteps".format(t+1))
		        tt.append(t)
		        break
		if isprint and t==step_limit:
			print("Episode finished after {} timesteps".format(t+1))
			tt.append(t) 
	return tt
