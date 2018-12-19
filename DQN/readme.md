# Deep Q learning for Cartplot

# Installation Dependencies:

* Python 2.7
* tensorflow 0.8 
* pygame
* matplotlib

# How to Run?

```
cd DQN
python q_learn.py -m "Run"
```

If you want to train the network from beginning, delete the saved_networks and run qlearn.py -m "Train"

# class and function explaining

* class DQN: the agent
** function create_Q_network: creat the Q network
** function create_training_method: the training method
** function train_Q_network: trains the network
** function play_game: playing the game

