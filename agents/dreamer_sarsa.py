#попытка воспроизвести дример - все детали, кроме автоэнкодера.
#вместо a2c сделать sarsa. a2c так-то довольно хорош, но необходимость в цикле крутить q-функцию - это треш, это очень долго.
#При обучении сарса отбирать из возможных траекторий наиболее профитные по каждому seed
import sys
import pylab
import random
import re
import numpy as np
import pandas as pd
from collections import deque
from sklearn.metrics import pairwise
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, GRU, TimeDistributed, Reshape, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
sys.path.append('./common/')
import networkx as nx
import itertools
import utils
import copy
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
from sklearn import linear_model
import sklearn

#Как обойти нестабильность: надо делать прогноз так: последовательность кадров + экшнов + будущих экшнов -> последовательность будущих кадров
#решения принимаем совершенно примитивно: перебираем несколько планов, берём наилучший.
#мы не пытаемся здесь сделать дример или ещё какую-нибудь хорошую модель - просто пусть будет хороший, надёжный model based

class DreamerAgent:
    def __init__(self, state_size, action_size, layers_size=[450,450],deque_len=2000):
        self.render = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = 0.99 #0.98**15 ~ 0.75
        self.learning_rate = 0.0001
        self.epsilon_decay = 0.992
        #0.6*0.996^1000 = 0.01, то есть за 1000 тактов перестанем рандомить, а за 600 тактов сведём рандом к 10% от исходного
        self.epsilon_min = 0.01
        self.batch_size = 290
        self.sub_batch_size=1000
        self.train_start = 1500
        self.learning_frequency = 0.0
        self.reward_part_need = 0.3
        self.planning_horison = 400
        self.elementary_range = 90#сколько кадров у нас на выходе
        self.curiosity = 0.15
        #self.curiosity = 0
        self.layers_size = layers_size
        self.deque_len = deque_len
        
        #скорость принятия решения на 100 и на 500:
        #100 40/3:0.877s
        #500:3.33s
        #500 и count_steps=1: 1.18s
        #100 с горизонтом 70/2: 0.68s
        self.is_fit = False
        self.q_fit = False
        
        self.count_trajectories = 30
        self.trajectory_seed_count=100
        self.trajectories_percentile=0.0
        self.count_steps = 6#очень связано с self.elementary_range и self.planning_horison. self.elementary_range*self.count_steps должно быть больше self.planning_horison, и желательно не на 1-2, а раза в 2.
        
        #110*(110*5-400)*50
        self.reset()
        
        
        
    def reset(self):
        # create replay memory using deque
        self.epsilon = 0.6
        self.s = deque(maxlen=self.deque_len)
        self.r = deque(maxlen=self.deque_len)
        #действия в формате one_hot:
        self.a = deque(maxlen=self.deque_len)
        self.done = deque(maxlen=self.deque_len)
        # SR модель - для оценки value
        self.model_sr = self.build_model('sr')
        # SS модель - для оценки будущего поведения мира независимо от меня
        #self.model_ss = self.build_model('ss')
        # SAS модель - для оценки advantage (нахрен advantage) будущего поведения мира, в зависимости от actions прошлых и будущих
        self.model_sas = self.build_model('sas')
        self.model_sq = self.build_model('sq')
        self.model_sa = self.build_model('sa')
        #инициализировать модельки
        self.train_model(epochs=1)
        
    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self,type_mdl):
        if type_mdl=='sr':
            input_dim = self.state_size 
            out_dim = 1
        
            model = Sequential()
            model.add(BatchNormalization())
            model.add(Dense(int(self.layers_size[0]), input_dim=input_dim, activation='relu',
                            kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))
            model.add(Dropout(rate=0.3))
            model.add(BatchNormalization())
            model.add(Dense(int(self.layers_size[1]), activation='relu',
                            kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))
            model.add(Dropout(rate=0.3))
            model.add(Dense(out_dim, activation='linear',
                            kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))
            model.compile(loss='mse', optimizer=Adam(lr=0.3*self.learning_rate))
        elif type_mdl=='sq':
            input_dim = self.state_size 
            out_dim = 1
        
            model = Sequential()
            model.add(BatchNormalization())
            model.add(Dense(int(self.layers_size[0]), input_dim=input_dim, activation='relu',
                            kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))
            model.add(Dropout(rate=0.3))
            model.add(BatchNormalization())
            model.add(Dense(int(self.layers_size[1]), activation='relu',
                            kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))
            model.add(Dropout(rate=0.3))
            model.add(Dense(out_dim, activation='linear',
                            kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate*5))
        elif type_mdl=='sa':
            #a - это advantage. Но это нафиг неважно
            input_dim = self.state_size 
            out_dim = self.action_size
        
            model = Sequential()
            model.add(BatchNormalization())
            model.add(Dense(int(self.layers_size[0]), input_dim=input_dim, activation='relu',
                            kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))
            model.add(Dropout(rate=0.3))
            model.add(BatchNormalization())
            model.add(Dense(int(self.layers_size[1]), activation='relu',
                            kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))
            model.add(Dropout(rate=0.3))
            model.add(Dense(out_dim, activation='linear',
                            kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        elif type_mdl=='sas':
            input_state = Input(shape=[self.elementary_range,self.state_size])
            input_action = Input(shape=[self.elementary_range,self.action_size])
            input_nxt_action = Input(shape=[self.elementary_range,self.action_size])
            x = input_state
            x = TimeDistributed(Dense(210, activation='selu',
                            kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)),name='x0')(x)
            x = TimeDistributed(Dense(210, activation='selu',
                            kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)),name='x1')(x)
            x = Flatten()(x)
            x = Dense(1200, activation='selu',
                            kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001),name='x2')(x)
            #
            x2 = TimeDistributed(Dense(16, activation='selu'),name='x3')(input_action)
            x2 = TimeDistributed(Dense(16, activation='selu'),name='x4')(x2)
            x2 = Flatten()(x2)
            x2 = Dense(300, activation='selu',
                            kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001))(x2)
            #
            x3 = TimeDistributed(Dense(16, activation='selu'),name='x3_')(input_nxt_action)
            x3 = TimeDistributed(Dense(16, activation='selu'),name='x4_')(x3)
            x3 = Flatten()(x3)
            x3 = Dense(380, activation='selu',
                            kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001))(x3)
            x = keras.layers.concatenate([x,x2,x3])
            # Bottleneck here!
            x = Dense(2000, name='bottleneck', activation='tanh')(x)
            # Start scaling back up
            # No frame stack for output
            cells_for_one_state = 90
            bottleneck_out_shape = self.elementary_range*cells_for_one_state
            x = Dense(bottleneck_out_shape, activation='relu')(x)
            x = Reshape((self.elementary_range,cells_for_one_state))(x)
            x = TimeDistributed(Dense(cells_for_one_state, activation='relu',
                        kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))(x)
            future = TimeDistributed(Dense(self.state_size, activation='relu',
                        kernel_initializer='he_uniform',kernel_regularizer=keras.regularizers.l2(0.0001)))(x)
            # this model maps an input to its reconstruction
            model = Model([input_state, input_action, input_nxt_action], future)

            model.compile(loss='mse',
                          optimizer=Adam(lr=self.learning_rate*0.1))
            model.summary()
   
        self.is_fit = False
        #model.summary()
        return model
    
    def generate_trajectories(self, count=10, steps=3,s=None, a=None, a_nxt=None, s_lst=None, a_lst=None):
        #сделать траектории на базе рандомных экшнов
        #s - тензор state_sizeXtime_len
        #a - тензор action_sizeXtime_len
        #s_lst - тензор countXstate_sizeXtime_len
        #a_lst - тензор countXaction_sizeXtime_len
        #a_nxt - тензор countXaction_sizeXtime_произвольной_длины
        if (a is None) and (a_lst is None):
            a = np.array(self.a)[-self.elementary_range:]
        if (s is None) and (s_lst is None):
            s = np.array(self.s)[-self.elementary_range:]

        write_a_nxt = a_nxt is None
        write_s_lst = s_lst is None
        write_a_lst = a_lst is None
        

        if write_a_nxt:
            a_nxt = []
        if write_a_lst:
            a_lst = []
        if write_s_lst:
            s_lst = []
        for i in range(count):
            if write_a_nxt:
                a_nxt_curr = np.array(np.round(np.random.rand(self.elementary_range)*(self.action_size-1)),dtype=int)
                one_hot_actions = np.zeros((self.elementary_range, self.action_size))
                one_hot_actions[range(self.elementary_range), a_nxt_curr] = 1
                a_nxt_curr = one_hot_actions
                a_nxt.append(a_nxt_curr)
            if write_a_lst:
                a_lst.append(a)
            if write_s_lst:
                s_lst.append(s)
        if write_a_nxt:
            a_nxt = np.array(a_nxt)

        a = np.array(a_lst)
        s = np.array(s_lst)
        if (not write_a_nxt) and (np.shape(a_nxt)[1]>self.elementary_range):
            a_nxt = a_nxt[:,:self.elementary_range]
            a_nxt4_nxt = a_nxt[:,self.elementary_range:]
            #можно подать на вход a_nxt не просто как план на 40 тактов вперёд, а как план на 160, например, тактов. 
            #В этом месте идёт нарезка - что подать в качестве a, а что a_nxt
        else:
            a_nxt4_nxt = None
        try:    
            s_nxt = self.model_sas.predict([s,a,a_nxt])
        except Exception:
            print(np.shape(s),np.shape(a),np.shape(a_nxt),np.shape(np.array(a_nxt)),'write_a_nxt',write_a_nxt)
            print('s',s)
            print('write_s_lst',write_s_lst)
            print('s_lst',s_lst)
            s_nxt = self.model_sas.predict([s,a,a_nxt])
        if steps>1:
            s_nxt_new,a_new,a_nxt_new = self.generate_trajectories(count=count,steps=steps-1, s_lst=s_nxt, a_lst=a_nxt, a_nxt=a_nxt4_nxt)
            s_nxt = np.concatenate([s_nxt,s_nxt_new],axis=1)
            a_nxt = np.concatenate([a_nxt,a_nxt_new],axis=1)
            a = np.concatenate([a,a_new],axis=1)
        return s_nxt,a_nxt,a
        
    
    def estimate_trajectories(self, s):
        #s - тензор countXtime_произвольной_длиныXstate_size
        s_new = np.concatenate(s,axis=0)
        #s_new = np.concatenate(np.concatenate(s_new,axis=0),axis=0)
        r_long = self.model_sr.predict(s_new)
        r = []
        for i in range(np.shape(s)[0]):
            start = i*np.shape(s)[1]
            end = (i+1)*np.shape(s)[1]
            r.append(r_long[start:end])
        shp = np.shape(r)
        return np.reshape(np.array(r),[shp[0],shp[1]])
        
    def get_action(self, state, verbose=0):
        if (np.random.rand() <= self.epsilon) or (len(self.s)<self.train_start) or (self.is_fit==False) or (self.q_fit==False):
            return random.randrange(self.action_size)
        else:
            action_one_hot = self.model_sa.predict(state)[0]
            if verbose:
                print('action_one_hot',np.round(action_one_hot,4),'action',np.argmax(action_one_hot))
            return np.argmax(action_one_hot)
    
    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        a_one_hot = np.zeros(self.action_size)
        a_one_hot[action]=1
        self.s.append(state[0,:])
        self.a.append(a_one_hot)
        self.r.append(reward)
        self.done.append(done)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def make_discounted_rewards(self):
        #derivative - брать ли за награду скорость движения к цели
        idx_borders = np.array(self.done)
        #borders - границы между эпизодами
        self.r_disco = utils.exp_smooth(self.r,self.discount_factor,self.planning_horison,idx_borders)
    
    #аугментация данных. Придётся выбросы размножить, потому что реворды очень на них завязаны
    #ну и данные выложим в виде (s,a,r_disco)
    def rebalance_data(self,s,action,reward):
        mean = np.mean(reward)
        #mean75 = (np.mean(reward)+np.percentile(reward,0.97))*0.5
        #mean25 = (np.mean(reward)+np.percentile(reward,0.03))*0.5
        mean75 = np.percentile(reward,75)
        mean25 = np.percentile(reward,25)
        if mean75==mean25:
            mean75 = np.mean(reward[reward>mean])
            mean25 = np.mean(reward[reward<mean])
        #размножь большие
        idx_big = reward>=mean75
        print('mean75',mean75)
        if any(idx_big):
            idx_big_num = np.where(idx_big)[0]
            s_add = list(s[idx_big_num])
            action_add = list(action[idx_big_num])
            reward_add = list(reward[idx_big_num])
    
            initial_part = np.mean(idx_big)
            multiplication_coef = int(self.reward_part_need/initial_part) - 1
    
            for i in range(multiplication_coef):
                s=np.vstack((s,s_add))
                action = np.concatenate((action,action_add))
                reward = np.concatenate((reward,reward_add))
        #размножь мелкие
        idx_small = reward<=mean25
        print('mean25',mean25)
        if any(idx_small):
            idx_small_num = np.where(idx_small)[0]
            s_add = list(s[idx_small_num])
            action_add = list(action[idx_small_num])
            reward_add = list(reward[idx_small_num])
    
            initial_part = np.mean(idx_small)
            multiplication_coef = int(self.reward_part_need/initial_part) - 1
            for i in range(multiplication_coef):
                s=np.vstack((s,s_add))
                action = np.concatenate((action,action_add))
                reward = np.concatenate((reward,reward_add))
        #аугментировать по экшнам. Сделать частоту каждого экшна в выборке >= 0.3/число экшнов
        freq_arr=np.zeros(self.action_size)
        for corrections_count in range(6):
            corrections_flag=False
            for i in range(self.action_size):
                freq_arr[i]=np.mean(action[:,i]==1)
                if freq_arr[i]<0.3/self.action_size and freq_arr[i]>0:
                    corrections_flag=True
                    #размножаем
                    idx_act_num = np.where(action[:,i]==1)[0]
                    s_add = list(s[idx_act_num])
                    action_add = list(action[idx_act_num])
                    reward_add = list(reward[idx_act_num])
                    s=np.vstack((s,s_add))
                    action = np.concatenate((action,action_add))
                    reward = np.concatenate((reward,reward_add))
            corrections_count+=1
            if not corrections_flag:
                break
        return (s,action,reward)
    
    def update_target_model(self):
        self.train_model(epochs=210,sub_batch_size=6000,verbose=False, draw=True)
        self.train_model(epochs=1,sub_batch_size=6000,verbose=False)
        self.dream()
        
    def test_model(self,model,X,Y,show_result=False):
        Y_pred = model.predict(X)
        mse = np.mean((Y_pred-Y)**2)
        if show_result:
            return mse, Y_pred
        else:
            return mse
   
    def get_batch(self, windowed_frames, windowed_actions, dones, batch_size, indices):
        while True:
            for i in indices:
                start = i * batch_size
                middle = int((i + 0.5) * batch_size)
                end = (i + 1) * batch_size
                if np.sum(dones[start:end])>0:
                    continue
                s = np.reshape(windowed_frames[start:middle],[int(batch_size/2),self.state_size,1])
                a = np.reshape(windowed_actions[start:middle],[int(batch_size/2),self.action_size,1])
                s_nxt = np.reshape(windowed_frames[middle:end],[1,int(batch_size/2),self.state_size])
                a_nxt = np.reshape(windowed_actions[middle:end],[int(batch_size/2),self.action_size,1])
                yield (
                    [s, a,a_nxt],
                    s_nxt
                )

    def train_model(self,epochs=4,sub_batch_size=None,verbose=False,screen_curious=False,draw=False,diaposon=[-800,-1]): 
        if len(self.s)<=self.elementary_range*(self.count_steps+2):
            return
        if len(self.s) < self.train_start:
            return
        if (epochs<5) and (np.random.rand()>self.learning_frequency):
            return
        print(f'train, draw={draw}')
        batch_size = self.elementary_range*2
        num_batches = len(self.s) // batch_size
        indices = np.random.permutation(num_batches)
        
        #обучить модель sas
        #нарубить батчи
        sx_batches = []
        sy_batches = []
        ax_batches = []
        axnxt_batches = []
        s = np.array(self.s)
        a = np.array(self.a)
        r = np.array(self.r)
        done = np.ravel(np.array(self.done))
        for i in np.arange(len(self.s)-self.elementary_range*2):
            i = int(i)
            done_batch = done[i:i+self.elementary_range*2]
            if np.sum(done_batch)==0:
                sx_batch = s[i:i+self.elementary_range]
                sx_batches.append(sx_batch)
                sy_batch = s[i+self.elementary_range:i+2*self.elementary_range]
                sy_batches.append(sy_batch)
                ax_batch = a[i:i+self.elementary_range]
                ax_batches.append(ax_batch)
                axnxt_batch = a[i+self.elementary_range:i+2*self.elementary_range]
                axnxt_batches.append(axnxt_batch)
        sx_batches = np.array(sx_batches)
        sy_batches = np.array(sy_batches)
        ax_batches = np.array(ax_batches)
        axnxt_batches = np.array(axnxt_batches)
        if draw:
            #расскажи про качество
            #x = [self.get_batch(np.array(self.s), np.array(self.a),np.ravel(np.array(self.done)),batch_size, indices) for x in range(1000)]
            x= [sx_batches,ax_batches,axnxt_batches]
            sy_batches_predict = self.model_sas.predict(x)
            rmstd = np.mean(np.abs(sy_batches_predict - sy_batches))/np.mean(np.std(sy_batches))
            print('rmstd sas test',rmstd)
            #multy-step прогноз
            steps = 3
            trials = 4
            for i in range(trials):
                for j in range(200):
                    idx = int(np.random.rand()*(len(self.s)-1-self.elementary_range*(steps+2)))
                    if np.sum(done[idx:idx+self.elementary_range*steps])==0:
                        break
                    else:
                        idx=-1
                if idx!=-1:
                    sx_batch = s[idx:idx+self.elementary_range]
                    ax_batch = a[idx:idx+self.elementary_range]
                    axnxt_batch = a[idx+self.elementary_range:idx+self.elementary_range*(steps+1)]
                    sy_batch = s[idx+self.elementary_range:idx+self.elementary_range*(steps+1)]
                    sx_batch = np.reshape(sx_batch, [1,np.shape(sx_batch)[0],np.shape(sx_batch)[1]])
                    ax_batch = np.reshape(ax_batch, [1,np.shape(ax_batch)[0],np.shape(ax_batch)[1]])
                    axnxt_batch = np.reshape(axnxt_batch, [1,np.shape(axnxt_batch)[0],np.shape(axnxt_batch)[1]])
                    #print(np.shape(sx_batch),np.shape(ax_batch),np.shape(axnxt_batch))
                    
                    #sy_batches_predict, a_new, a_nxt_new = self.generate_trajectories(count=1, steps=steps,s=sx_batch, a=ax_batch, a_nxt=axnxt_batch, s_lst=None, a_lst=None)
                    #рекурсивный прогноз
                    sy_pred_lst = []
                    for j in range(steps):
                        axnxt_batch_loc = axnxt_batch[:,j*self.elementary_range:(j+1)*self.elementary_range]
                        x= [sx_batch,ax_batch,axnxt_batch_loc]
                        try:
                            sx_batch=self.model_sas.predict(x)
                        except Exception:
                            print("sx_batch=self.model_sas.predict(x) Error")
                            return
                        sy_pred_lst.append(sx_batch)
                        ax_batch = axnxt_batch_loc
                    sy_batches_predict = np.concatenate(sy_pred_lst,axis=0)
                    sy_batches_predict = np.concatenate(sy_batches_predict,axis=0)
                    sy_batch_predict = sy_batches_predict[0]
                    error = sy_batch_predict-sy_batch
                    mae = np.mean(np.abs(error),axis=1)
                    mae_std = mae/np.std(sy_batch)
                    plt.title(f'from {idx}, {steps} steps')
                    plt.plot(mae_std)
                    plt.show()
        #self.model_sas.fit((sx_batches,ax_batches,axnxt_batch),sy_batches,)
        validation_split = 0.15
        #early_stopping = EarlyStopping(patience=3)
        prev_value = -1
        t = pd.Timestamp.now()
        for i in range(6):
            print('fitting sas...')
            self.model_sas.fit(x=self.get_batch(
                np.array(self.s), np.array(self.a),np.ravel(np.array(self.done)), 
                batch_size, indices[:int((1-validation_split)*len(indices))]),
                steps_per_epoch=int(len(indices)),
                epochs=int(epochs*1),
                verbose=False)
            x= [sx_batches,ax_batches,axnxt_batches]
            sy_batches_predict = self.model_sas.predict(x)
            rmstd = np.mean(np.abs(sy_batches_predict - sy_batches))/np.mean(np.std(sy_batches))
            if prev_value<0:
                prev_value = rmstd
            if prev_value<rmstd:
                break
        print(pd.Timestamp.now()-t)
        if draw:
            #расскажи про качество
            #x = [self.get_batch(np.array(self.s), np.array(self.a),np.ravel(np.array(self.done)),batch_size, indices) for x in range(1000)]
            x= [sx_batches,ax_batches,axnxt_batches]
            sy_batches_predict = self.model_sas.predict(x)
            rmstd = np.mean(np.abs(sy_batches_predict - sy_batches))/np.mean(np.std(sy_batches))
            print('rmstd sas train',rmstd)
        
        #обучить модель sr
        #эти сдвиги нужны, потому что реально происходит так: state, при нём делается action, а после получается reward.
        (s,a,r) = self.rebalance_data(s[1:],a[1:],r[:-1])
        if np.std(r)==0:
            #Нечего тут учить
            print('std(r)=0')
            plt.plot(np.ravel(self.r))
            plt.show()
            return
        else:
            self.model_sr.fit(s, r, batch_size=6096,epochs=epochs, verbose=False)
            mse = self.test_model(self.model_sr,s,np.array(np.ravel(r),ndmin=2,dtype=np.float32).T)
            if epochs>5:
                print('sr rmse', np.sqrt(mse),'sr rmse/std', np.sqrt(mse)/np.std(r), 'sr rmse/mean', np.sqrt(mse)/np.mean(np.abs(r)), 's.shape', np.shape(s))
            if draw:
                #пересчитаем всё заново
                r_sr_predicted_local = self.model_sr.predict(np.array(self.s))
                print('sr fact, frc')
                plt.plot(np.ravel(np.array(self.r)[diaposon[0]:diaposon[1]]))
                plt.plot(np.ravel(r_sr_predicted_local[diaposon[0]:diaposon[1]]))
                plt.show()
        self.is_fit = True

        
    def dream(self,diaposon=[-600,-1]):
        if not self.is_fit:
            print('dont dream: not fit')
            return
        #создать кучу траекторий
        #сделать таблицу sarn, где n - next state
        #q(s) = 0, затем q(s) = reward(s) + disco*q(n), повторять рекурсивно
        #чтобы проверить адекватность, взять реальную историю и построить q(s) на фоне ревордов. 
        #Должно получиться что-то вроде r_disco
        
        #создать кучу траекторий
        t = pd.Timestamp.now()
        print('dreaming')
        if len(self.s)<=self.elementary_range*(self.count_steps+2):
            print('not enough data')
            return
        s_lst = []
        a_lst = []
        done = np.ravel(np.array(self.done))
        for i in range(self.trajectory_seed_count):
            for j in range(300):
                #создать seed
                idx = int(np.random.rand()*(len(self.s)-1-int(self.elementary_range*1.1)))
                if np.sum(done[idx:idx+self.elementary_range])==0:
                    break
                else:
                    idx=-1
            if idx>0:
                #print('idx',idx,np.shape(np.array(self.s)[idx:idx+self.elementary_range]))
                s_nxt,a_nxt,a = self.generate_trajectories(count=self.count_trajectories, steps=self.count_steps,s=np.array(self.s)[idx:idx+self.elementary_range], a=np.array(self.a)[idx:idx+self.elementary_range])
                r=self.model_sr.predict(np.concatenate(s_nxt,axis=0), batch_size=4096)
                r=np.reshape(r,(np.shape(s_nxt)[0],np.shape(s_nxt)[1]))
                #отобрать наилучшие траектории из каждого сида
                r_sums = np.sum(r,axis=1)
                idx_best = r_sums>=np.percentile(r_sums,self.trajectories_percentile*100)
                s_nxt = s_nxt[idx_best]
                a_nxt = a_nxt[idx_best]
                s_lst.append(s_nxt)
                a_lst.append(a_nxt)
        if len(s_lst)==0:
            print(f'could not make seeds for trajectories, s size: {len(self.s)}, free_space_needed: {int(self.elementary_range*1.1)}, dones: {np.sum(done)}')
            return
        print('s_nxt size',np.shape(s_nxt),'s_lst count',len(s_lst))
        s_lst = np.concatenate(s_lst,axis=0)
        a_lst = np.concatenate(a_lst,axis=0)
        #s_lst - тензор countXtime_lenXstate_size
        #a_lst - тензор countXtime_lenXa_size
        
        s=np.concatenate(s_lst)
        d_lst = np.zeros( (np.shape(s_lst)[0],np.shape(s_lst)[1]) )
        d_lst[:,-1] = 1
        d=np.ravel(d_lst)
        
        print(f'generated states raw:{np.shape(s)}, std(s[:,1]):{np.std(s[:,1])}')
        r=self.model_sr.predict(s, batch_size=4096)
        print(f'rewards:{(np.min(r),np.median(r),np.max(r))}')
        #Рассчитать r_disco для всех траекторий
        r_disco = utils.exp_smooth(r,self.discount_factor,self.planning_horison,d)
        #убрать те части траекторий, где абберация
        r_disco_lst = np.reshape(r_disco,[np.shape(a_lst)[0],np.shape(a_lst)[1]])
        s_lst = s_lst[:,:-self.planning_horison]
        a_lst = a_lst[:,:-self.planning_horison]
        r_disco_lst = r_disco_lst[:,:-self.planning_horison]
        #снова всё сделать плоско
        s = np.vstack(s_lst)
        a = np.vstack(a_lst)
        r_disco = np.ravel(r_disco_lst)
        q = r_disco
        print(f'generated states no abberation:{np.shape(s)}, std(s[:,1]):{np.std(s[:,1])}, {pd.Timestamp.now()-t}')
        t = pd.Timestamp.now()
        #добалансировать
        mean = np.mean(q)
        mean75 = np.mean(q[q>mean])
        mean25 = np.mean(q[q<mean])
        #размножь большие
        idx_big = np.ravel(q)>=mean75
        if any(idx_big):
            idx_big_num = np.where(idx_big)[0]
            s_add = list(s[idx_big_num])
            a_add = list(a[idx_big_num])
            q_add = list(q[idx_big_num])
            initial_part = np.mean(idx_big)
            multiplication_coef = int(self.reward_part_need/initial_part) - 1
            for i in range(multiplication_coef):
                s=np.vstack((s,s_add))
                a=np.vstack((a,a_add))
                q = np.concatenate((q,q_add))
            print(f'q_mult_coef big {multiplication_coef}')
        #размножь мелкие
        idx_small = np.ravel(q)<=mean25
        if any(idx_small):
            idx_small_num = np.where(idx_small)[0]
            s_add = list(s[idx_small_num])
            a_add = list(a[idx_small_num])
            q_add = list(q[idx_small_num])
            initial_part = np.mean(idx_small)
            multiplication_coef = int(self.reward_part_need/initial_part) - 1
            print(f'q_mult_coef small {multiplication_coef}')
            for i in range(multiplication_coef):
                s=np.vstack((s,s_add))
                a=np.vstack((a,a_add))
                q = np.concatenate((q,q_add)) 
        ##
        print(f'generated states balanced:{np.shape(s)}, std(s[:,1]):{np.std(s[:,1])}')
        self.model_sq.fit(s,q,
                    epochs=50,
                    batch_size=8192*2,
                    verbose=False)
        r_disco_pred = self.model_sq.predict(s)
        print(f'q ready {pd.Timestamp.now()-t}')
        t = pd.Timestamp.now()
        #advantage
        advantage = np.zeros((np.shape(s)[0], self.action_size))
        r_disco = np.reshape(q,[np.shape(s)[0],1])
        r_disco_pred = np.reshape(r_disco_pred,[np.shape(s)[0],1])
        advantage += a*(r_disco - r_disco_pred)
        if np.sum(a!=0)==0:
            print('a==0')
        if np.sum(q!=0)==0:
            print('q==0') 
        print('advantage',advantage)
        self.model_sa.fit(s,advantage,epochs=50,
                    batch_size=8192*2,
                    verbose=False)
        print('model_sa fit')
        q_pred = self.model_sq.predict(np.array(self.s), batch_size=2048)
        if np.std(q_pred[diaposon[0]:diaposon[1]])>0:
            coef = np.abs(np.std(np.array(self.r)[diaposon[0]:diaposon[1]])/np.std(q_pred[diaposon[0]:diaposon[1]]))
        else:
            coef = 1
        plt.title(f'r, q_predicted, coef={np.round(coef,4)}')
        plt.plot(np.ravel(self.r)[diaposon[0]:diaposon[1]])
        plt.plot(coef*np.ravel(q_pred)[diaposon[0]:diaposon[1]])
        plt.show()