# Solver for discrete environments from OpenAI Gym

[OpenAI gym](link)

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

## How to use

### Learning algorithm

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
