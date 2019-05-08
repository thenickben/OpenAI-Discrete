# Solver for discrete environments from OpenAI Gym

[OpenAI gym](https://gym.openai.com/)

## How to install and run

### 1) Clone this repo and make sure to install all the requirements
```
$ git clone https://github.com/thenickben/OpenAI-Discrete
$ pip install -r requirements.txt
```

### 2) Training the agent

Requires `Python 3`
```
$ python main.py
```
(this will train for the default configuration, but it can be used as a quick test)

## How to use

### Environment

Simply pass the environment name:
```
$ python main.py --env_name='FrozenLake-v0'
```
(default environment is 'Taxi-v2')

### Learning algorithms

Currently the following action-value function learning algorithms are supported (more to come!):

* Q-Learning
* Sarsa
* Expected Sarsa
* Double Q-Learning
* Double Sarsa
* Double Expected Sarsa

There is an option to pass which algorithm the agent will follow. For example, for Sarsa:
```
$ python main.py --learning_algo='sarsa'
```

There is also the option to train agents using all the supported algorithms and plot the results at the end:
```
$ python main.py --run_all='True' --plot="True"
```
### Other options

Additional options and their defaults are:

```
--episodes', default = 10000
--alpha', default = 0.1
--gamma', default = 1.0
--plot', default = 'False'
--smoothing_window', default = 100
```

## Results

### Taxi-v2

Obtained best avg reward = 9.75 in approx 62k episodes with:
```
$ python main.py --learning_algo='sarsa' --plot="True" --episodes=100000 --smoothing_window=100 --alpha=0.01
```

Here is the plot when --run_all="True":

![Runs for OpenAI-gym-Taxi-v2 with different learning algorithms](https://github.com/thenickben/OpenAI-Discrete/blob/master/docs/run_all.png)