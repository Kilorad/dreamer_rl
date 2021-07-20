import gym
import sys
import pylab
import random
import numpy as np
from collections import deque
import keras
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
sys.path.append('./common/')
import utils


class randomAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.action_size = action_size
        self.memory = []
        self.epsilon = 1

    # get action from model using epsilon-greedy policy
    def get_action(self, state,verbose=False):
        return random.randrange(self.action_size)
    def append_sample(self,a,b,c,d,e):
        pass
    def train_model(self,epochs=None,sub_batch_size=None,screen_curious=None):
        pass
    def update_target_model(self):
        pass
    def reset(self):
        pass