import sys
import pylab
import random
import re
import numpy as np
import pandas as pd
from collections import deque
from sklearn.metrics import pairwise
import os
import psutil
import random 

import matplotlib.pyplot as plt
sys.path.append('./common/')
import networkx as nx
import itertools
import utils
import copy

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D,Conv3D,Conv2D,TimeDistributed,Flatten, Reshape
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, Permute, MaxPooling2D, MaxPooling3D, UpSampling2D
from keras.layers import LeakyReLU
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import GRU, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K

import pickle


_IMAGE_NET_TARGET_SIZE = (224, 224)
MAX_LENGTH = 40


class AutoencoderRLWrapper:
    def __init__(self, state_size, action_size, encoder_layers_count, path_to_network, x_path,y_path, p_add_to_learn=0.0, learn_size=500, x_len=6,y_len=6, period=8,learn_size_raw=100,learn_size_preprocessed=1000,root=''):
        #на входе в автоэнкодер: последовательность из x_len кадров (разный 8-й, если period=8), на выходе - последний из y_len-последовательности
        self.root=root
        self.render = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        
        # create replay memory using deque
        self.period = period
        deque_size = (x_len + y_len)*period#*15
        self.s = deque(maxlen=deque_size)
        self.done = deque(maxlen=deque_size)
        
        #учебная память
        self.x_len = x_len
        self.y_len = y_len
        self.p_add_to_learn = p_add_to_learn #вероятность того, что мы конец self.s запишем в учебную память
        self.learn_size = learn_size
        self.learn_sequences_raw = deque(maxlen=learn_size_raw)
        self.learn_sequences_preprocessed = deque(maxlen=learn_size_preprocessed)
        
        #
        self.encoder_layers_count = encoder_layers_count
        self.path_to_network = path_to_network
        #Это должен быть list!
        self.x_path = x_path
        self.y_path = y_path
        if os.path.isfile(self.path_to_network):
            self.autoencoder_network = self.make_main_autoencoder_network()
            self.autoencoder_network.load_weights(self.path_to_network)
            layer_name = self.autoencoder_network.layers[37].name
            for i in range(30,60):
                print(i,self.autoencoder_network.layers[i].name)
            self.encoder_network = Model(inputs=self.autoencoder_network.input, 
                                              outputs=self.autoencoder_network.get_layer(layer_name).output)
        
    def reset_keras(self):
        K.clear_session()       
    def decode(self, embedding):
        layer_name = self.autoencoder_network.layers[x].name
        layer_name_end = self.autoencoder_network.layers[-1].name
        decoder_network = Model(inputs=self.autoencoder_network.get_layer(layer_name).input, 
                                              outputs=self.autoencoder_network.get_layer(layer_name_end).output)
        s_decoded = decoder_network.predict(embedding)
        return s_decoded    
    def make_main_autoencoder_network(self):
        kullback_leibler_divergence = keras.losses.kullback_leibler_divergence
        K = keras.backend

        def kl_divergence_regularizer(inputs):
            means = K.mean(inputs, axis=0)
            return 0.005 * (kullback_leibler_divergence(0.05, means)
                         + kullback_leibler_divergence(1 - 0.05, 1 - means))

        xlen = 6
        frames_out = 1
        x_subseq_len = xlen
        droprate = 0.085
        model = Sequential()
        filters3d = 30
        model = Sequential()
        model.add(BatchNormalization())
        model.add(Conv3D(filters3d, (1,3,3), padding='same', strides=(1,1,1), kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(LeakyReLU(alpha=0.05))
        model.add(MaxPooling3D(pool_size=(1, 4, 4), padding='same', strides = (1,2,2)))
        model.add(Dropout(droprate))
        #6x112x112x3
        filters3d = 50
        depth = 2
        model.add(BatchNormalization())
        model.add(Conv3D(filters3d, (3,3,3), padding='same', strides=(1,1,1), kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(LeakyReLU(alpha=0.05))
        model.add(MaxPooling3D(pool_size=(1, 4, 4), padding='same', strides = 2))
        model.add(Dropout(droprate))
        #3x56x56x72
        filters3d = 100
        model.add(BatchNormalization())
        model.add(Conv3D(filters3d, (2,4,4), padding='same', strides=(1,1,1), kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(LeakyReLU(alpha=0.05))
        model.add(MaxPooling3D(pool_size=(1, 4, 4), padding='same', strides = (2,2,2)))
        model.add(Dropout(droprate))
        #2x28x28x100
        filters3d = 90
        model.add(BatchNormalization())
        model.add(Conv3D(filters3d, (2,3,3), padding='same', strides=(1,1,1), kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(droprate))
        #2x28x28x100
        model.add(Reshape((28,28,2*filters3d)))
        #28x28x200
        model.add(BatchNormalization())
        model.add(Conv2D(120, (6, 6), padding='same', strides = 1, kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(droprate))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides = 2))
        #14x14x170
        model.add(BatchNormalization())
        model.add(Conv2D(190, (5, 5), padding='same', strides = 1, kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(droprate))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides = 2))
        #7x7x190
        embedding_cores = 2
        model.add(BatchNormalization())
        model.add(Conv2D(embedding_cores, (5, 5), padding='same', strides = 1, kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(droprate))

        embedding_size = embedding_cores*49
        embedding_size = 30

        model.add(Reshape((embedding_cores*49,)))
        model.add(BatchNormalization())
        model.add(Dense(embedding_size,kernel_initializer='he_uniform',kernel_regularizer=kl_divergence_regularizer))
        model.add(LeakyReLU(alpha=0.05))

        model.add(BatchNormalization())
        model.add(Dense(embedding_cores*49,kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(droprate))
        model.add(Reshape((7,7,embedding_cores)))

        filters = 390
        kernel_size = (7,7)
        strides = (1,1)
        model.add(BatchNormalization())
        model.add(Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(droprate))
        #
        #теперь развернуть в 3d, но не в 3д-матрицу, а time_disributed
        model.add(Reshape((frames_out,7,7,int(filters/frames_out))))

        filters = 460
        kernel_size = (7,7)
        strides = (1,1)
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Conv2DTranspose(input_shape=(7,7,x_subseq_len*2),filters=filters, kernel_size=kernel_size, strides=strides, padding='same',kernel_regularizer=keras.regularizers.l2(0.0001))))
        model.add(Dropout(droprate))
        model.add(LeakyReLU(alpha=0.05)) 

        filters = 350
        kernel_size = (7,7)
        strides = (1,1)
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',kernel_regularizer=keras.regularizers.l2(0.0001))))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(droprate))

        model.add(TimeDistributed(UpSampling2D((2, 2))))

        filters = 300
        kernel_size = (5,5)
        strides = (2,2)
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',kernel_regularizer=keras.regularizers.l2(0.0001))))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(droprate))

        #model.add(TimeDistributed(UpSampling2D((2, 2))))

        filters = 190
        kernel_size = (5,5)
        strides = (3,3)
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',kernel_regularizer=keras.regularizers.l2(0.0001))))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(droprate))

        filters = 180
        kernel_size = (7,7)
        strides = (1,1)
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',kernel_regularizer=keras.regularizers.l2(0.0001))))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(droprate))

        filters = 11
        kernel_size = (1,1)
        strides = (1,1)
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',kernel_regularizer=keras.regularizers.l2(0.0001))))
        model.add(LeakyReLU(alpha=0.05))

        lr = 0.00000001
        model.compile(loss=keras.losses.mean_squared_error,
                    optimizer=Adam(learning_rate=lr, clipnorm=lr*2))
  

        with open(self.root + self.x_path[0], 'rb') as f:
            x_movies = pickle.load(f)
        with open(self.root + self.y_path[0], 'rb') as f:
            y_movies = pickle.load(f)
        self.reset_keras()
        x_movies = x_movies[-2:]
        y_movies = y_movies[-2:]
        model.fit(
            x_movies,
            y_movies,
            epochs=1,
            verbose=2,
            batch_size=1
        )
        return model
        
    def make_embedding(self, state_arr,ravel=True, showtime=False):
        x_mask = [0, 4, 12, 24, 60, 105]
        
        if showtime:
            start = pd.Timestamp.now()
        #сделаю эмбеддинг энкодером
        #если не тот размер картинки, то отмасштабирую и докрашу чёрным лишнее
        #print("np.arange(-self.x_len*self.period,0,self.period)", np.arange(-self.x_len*self.period,0,self.period))
        def get_x_masked(x_pointer,pointer_now):
            #вывести картинку, которое сдвинуто относительно текущего указателя на столько-то. 
            #Оно может оказаться за пределами видеоряда, это нормально!
            if pointer_now-x_pointer>=0:
                additional_image = state_arr[pointer_now-x_pointer]
                #отмасштабировать, если надо
                shp = np.shape(additional_image)
                if (shp[0]!=_IMAGE_NET_TARGET_SIZE[0])or(shp[1]!=_IMAGE_NET_TARGET_SIZE[1]):
                    delta_x = int((_IMAGE_NET_TARGET_SIZE[0]- shp[0])/2)
                    delta_y = int((_IMAGE_NET_TARGET_SIZE[1]- shp[1])/2)
                    img = np.zeros((_IMAGE_NET_TARGET_SIZE[0],_IMAGE_NET_TARGET_SIZE[1],3))#чёрный квадрат
                    img[delta_x:delta_x+shp[0],delta_y:delta_y+shp[1],:] = additional_image#поверх рисуем кадр
                    additional_image = img
            else:
                
                additional_image = np.zeros((_IMAGE_NET_TARGET_SIZE[0],_IMAGE_NET_TARGET_SIZE[1],3))#чёрный квадрат
            return additional_image
        sequence_for_embedding = []
        for x_pointer_rel in x_mask:
            sequence_for_embedding.append(get_x_masked(x_pointer_rel,len(state_arr)-1))
        embedding = self.encoder_network.predict(np.array(sequence_for_embedding,ndmin=5))
        if ravel:
            shp = np.shape(embedding)
            embedding = np.reshape(embedding,(shp[0],-1))
        if showtime:
            print(pd.Timestamp.now() - start)
        return np.array(embedding,np.float32)

    def add_deeper_rl(self,rl_agent_object):
        self.deeper_rl = rl_agent_object
        self.deeper_rl.reset() #эта функция должна быть у любого rl
    
    def get_action(self, state, verbose=False):
        embedding = self.make_embedding(self.s)
        return self.deeper_rl.get_action(embedding,verbose)
    
    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.s.append(state)
        self.done.append(done)
        embedding = self.make_embedding(self.s)
        self.deeper_rl.append_sample(embedding, action, reward, next_state, done)
        if np.random.rand()<self.p_add_to_learn:
            #записать последовательность в self.learn_sequences_raw
            self.learn_sequences_raw.append()
    
    def update_target_model(self):
        #self.train_model(epochs=120,sub_batch_size=6000,verbose=0)
        self.deeper_rl.update_target_model()
        
    def test_model(self,model,X,Y,show_result=False):
        Y_pred = model.predict(X)
        mse = np.mean((Y_pred-Y)**2)
        if show_result:
            return mse, Y_pred
        else:
            return mse
    def make_screenshot(self, frame_number=-1, filename=None):
        if filename is None:
            plt.imshow(self.s[frame_number])
            plt.show()
            
    def train_model(self,epochs=4,sub_batch_size=None,verbose=False,screen_curious=True):
        #значит дообучаем низовую модель
        self.deeper_rl.train_model(epochs=epochs, sub_batch_size=sub_batch_size,screen_curious=screen_curious)
