# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:07:43 2018

@author: admin
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy


class cardRedBlack(object):
    def __init__(self):
        self.MonteCarloIterRound = 1000000
        self.TDLambdaIterRound = 1000
        self.faIterRound = 1000
        self.MonteCarloN0 = 100
        self.actionList = ['hit', 'stick']
        self.dealerSplit = [[1,4],[4,7],[7,10]]
        self.playerSplit = [[1,6],[4,9],[7,12],[10,15],[13,18],[16,21]]
    
    def stateInitialize(self):
        self.state = dict()
        self.state['dealer'] = np.random.randint(1, 11)
        self.state['player'] = np.random.randint(1, 11)
        return 'unfinished',0
        
    def step(self, action):
        if action == 'hit':
            self.state['player'] += self.cardpoolSimulator()
            if self.state['player'] < 1 or self.state['player'] > 21:
                return 'finished', -1
            else:
                return 'unfinished', 0
        else:
            dealerTurn = True
            while dealerTurn == True:
                self.state['dealer'] += self.cardpoolSimulator()
                if self.state['dealer'] < 1 or self.state['dealer'] > 21:
                    return 'finished', 1
                elif self.state['dealer'] >= 17:
                    if self.state['dealer'] > self.state['player']:
                        return 'finished', -1
                    elif self.state['dealer'] == self.state['player']:
                        return 'finished', 0
                    else:
                        return 'finished', 1
    
    def cardpoolSimulator(self):
        rand = np.random.randint(1, 4)
        if rand == 1:
            return (-1) * np.random.randint(1, 11)
        else:
            return np.random.randint(1, 11)
            
    def MonteCarloControl(self):
        self.MonteCarloStatsInitialize()
        self.rewardPrint = 0
        for _ in range(self.MonteCarloIterRound):
            self.MonteCarloOneEpisode()
            if _ % 10000 == 0:
                print(_, self.rewardPrint)
                self.rewardPrint = 0
        self.MonteCarloResultPlot()
        
    
    def MonteCarloStatsInitialize(self):
        self.stateVisits = dict()
        self.stateactionVisits = dict()
        self.q = dict()
        for i in range(1, 22):
            for j in range(1, 11):
                self.stateVisits[str(i)+'+'+str(j)] = 0
                self.stateactionVisits[str(i)+'+'+str(j)] = dict()
                self.stateactionVisits[str(i)+'+'+str(j)]['stick'] = 0
                self.stateactionVisits[str(i)+'+'+str(j)]['hit'] = 0
                self.q[str(i)+'+'+str(j)] = dict()
                self.q[str(i)+'+'+str(j)]['stick'] = 0 
                self.q[str(i)+'+'+str(j)]['hit'] = 0

    def MonteCarloOneEpisode(self):
        termination, reward = self.stateInitialize()
        pairStateAction = []
        while termination == 'unfinished':
            statePlayer = self.state['player']
            stateDealer = self.state['dealer']
            state = str(statePlayer) + '+' + str(stateDealer)
            self.stateVisits[str(statePlayer) + '+' + str(stateDealer)] += 1
            epsilon = self.MonteCarloN0 * 1.0 / (self.MonteCarloN0 + self.stateVisits[state])
            action = self.MonteCarloTakeAction(epsilon, state)
            self.stateactionVisits[str(statePlayer) + '+' + str(stateDealer)][action] += 1
            pairStateAction.append([str(statePlayer) + '+' + str(stateDealer), action])
            termination, reward = self.step(action)
        for pair in pairStateAction:
            self.q[pair[0]][pair[1]] += 1.0 / self.stateactionVisits[pair[0]][pair[1]] * reward
        self.rewardPrint += reward
        
    def BackTDLambdaOneEpisode(self):
        E = dict()
        termination, reward = self.stateInitialize()
        statePlayer = self.state['player']
        stateDealer = self.state['dealer']
        state = str(statePlayer) + '+' + str(stateDealer)
        action = self.HighLevelTakeAction(state)
        while termination == 'unfinished':
            termination, reward = self.step(action)        
            if termination != 'finished':      
                statePlayerNext, stateDealerNext = self.state['player'], self.state['dealer']
                stateNext = str(statePlayerNext) + '+' + str(stateDealerNext)
                actionNext = self.HighLevelTakeAction(stateNext)
                delta = reward + cl.q[stateNext][actionNext] - cl.q[state][action]
            else:
                delta = reward - cl.q[state][action]
            if state + '_' + action in list(E.keys()):
                E[state + '_' + action] += 1
            else:
                E[state + '_' + action] = 1
            for pairStateAction in list(E.keys()):
                state, action = pairStateAction.split('_')
                cl.q[state][action] += 1.0 / self.stateactionVisits[state][action] * delta * E[pairStateAction]
                E[pairStateAction] *= self.Lambda
            if termination != 'finished':
                state = stateNext
                action = actionNext
        self.rewardPrint += reward
                
    def faOneEpisode(self):
        self.E = np.zeros([36, 1])
        termination, reward = self.stateInitialize()
        statePlayer = self.state['player']
        stateDealer = self.state['dealer']
        state = str(statePlayer) + '+' + str(stateDealer)
        action = self.faTakeAction(state)
        feature = self.featureEncode(stateDealer, statePlayer, action)
        while termination == 'unfinished':
            termination, reward = self.step(action)
            if termination != 'finished':
                statePlayerNext, stateDealerNext = self.state['player'], self.state['dealer']
                stateNext = str(statePlayerNext) + '+' + str(stateDealerNext)
                actionNext = self.faTakeAction(stateNext)
                featureNext = self.featureEncode(stateDealerNext, statePlayerNext, actionNext)
                delta = reward + np.dot(self.w.T, featureNext)[0,0] - np.dot(self.w.T, feature)[0,0]
            else:
                delta = reward - np.dot(self.w.T, feature)[0,0]
            self.E = self.Lambda * self.E + feature
            wDelta = self.alpha * delta * self.E
            self.w += wDelta
            if termination != 'finished':
                state = stateNext
                action = actionNext
                feature = featureNext


                
    def faTakeAction(self, state):
        rand0 = np.random.rand(1)[0]
        if rand0 < self.epsilon:
            rand1 = np.random.randint(2)
            action = self.actionList[rand1]
        else:
            statePlayer, stateDealer = state.split('+')
            statePlayer , stateDealer = int(statePlayer), int(stateDealer)
            featureHit = self.featureEncode(stateDealer, statePlayer, 'hit')
            featureStick = self.featureEncode(stateDealer, statePlayer, 'stick')
            valueHit = np.dot(self.w.T, featureHit)[0,0]
            valueStick = np.dot(self.w.T, featureStick)[0,0]
            if valueHit > valueStick:
                action = 'hit'
            elif valueHit < valueStick:
                action = 'stick'
            else:
                rand1 = np.random.randint(2)
                action = self.actionList[rand1]
        return action
            
                
        
    def HighLevelTakeAction(self, state):
        self.stateVisits[state] += 1
        epsilon = self.MonteCarloN0 * 1.0 / (self.MonteCarloN0 + self.stateVisits[state])
        action = self.MonteCarloTakeAction(epsilon, state)
        self.stateactionVisits[state][action] += 1
        return action
    

    def MonteCarloTakeAction(self, epsilon, state):
        rand0 = np.random.rand(1)[0]
        if rand0 < epsilon:
            rand1 = np.random.randint(2)
            action = self.actionList[rand1]
        else:
            if self.q[state]['hit'] > self.q[state]['stick']:
                action = 'hit'
            elif self.q[state]['hit'] < self.q[state]['stick']:
                action = 'stick'
            else:
                rand1 = np.random.randint(2)
                action = self.actionList[rand1]
        return action   
        
    def MonteCarloResultPlot(self):
        xMatrix, yMatrix, zMatrix = np.zeros([21, 10]), np.zeros([21, 10]), np.zeros([21, 10])
        for i in range(1, 11):
            for j in range(1, 22):
                qHit = self.q[str(j) + '+' + str(i)]['hit']
                qStick = self.q[str(j) + '+' + str(i)]['stick']
                z = np.max([qHit, qStick])
                xMatrix[j-1, i-1] = i
                yMatrix[j-1, i-1] = j
                zMatrix[j-1, i-1] = z
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(xMatrix, yMatrix, zMatrix)
        plt.show()
        
    def BackTDLambdaControl(self, Lambda):
        self.Lambda = Lambda
        self.MonteCarloStatsInitialize()
        self.rewardPrint = 0
        for _ in range(self.TDLambdaIterRound):
            self.BackTDLambdaOneEpisode()
            if _ % 10000 == 0:
                print(_, self.rewardPrint)
                self.rewardPrint = 0
                
    def TDCompare(self, MCtrained = False):
        self.LambdaPool = np.arange(11) / 10
        mseList = []
        if MCtrained == False:
            self.MonteCarloControl()
        self.qMonteCarlo = copy.deepcopy(self.q)
        fig, axe = plt.subplots(2, 1, figsize = (8, 6))
        axe[0].set_xlabel('Lambda')
        axe[0].set_ylabel('MSE')
        for Lambda in self.LambdaPool:
            self.BackTDLambdaControl(Lambda)
            mse = self.mseStateActionCompute(self.qMonteCarlo, self.q)
            mseList.append(mse)
        msePool = np.array(mseList)
        axe[0].plot(self.LambdaPool, msePool)
        x0, mse0 = self.mseStepCurve(0.0)
        x1, mse1 = self.mseStepCurve(1.0)
        axe[1].plot(x0, mse0)
        axe[1].plot(x1, mse1)
        axe[1].legend(['lambda0', 'lambda1'])
        plt.tight_layout()
        
    def TDCompareFA(self, MCtrained = False):
        self.LambdaPool = np.arange(11) / 10
        mseList = []
        if MCtrained == False:
            self.MonteCarloControl()
        self.qMonteCarlo = copy.deepcopy(self.q)
        fig, axe = plt.subplots(2, 1, figsize = (8, 6))
        axe[0].set_xlabel('Lambda')
        axe[0].set_ylabel('MSE')
        for Lambda in self.LambdaPool:
            print(Lambda)
            self.faTDLambdaControl(Lambda)
            mse = self.mseStateActionComputeFA(self.qMonteCarlo, self.w)
            mseList.append(mse)
        msePool = np.array(mseList)
        axe[0].plot(self.LambdaPool, msePool)
        x0, mse0 = self.mseStepCurveFA(0.0)
        x1, mse1 = self.mseStepCurveFA(1.0)
        axe[1].plot(x0, mse0)
        axe[1].plot(x1, mse1)
        axe[1].legend(['lambda0', 'lambda1'])
        plt.tight_layout()
        
    
        
        
    def mseStepCurve(self, Lambda):
        self.Lambda = Lambda
        self.MonteCarloStatsInitialize()
        mseList = []
        xList = []
        for i in range(50000):
            self.BackTDLambdaOneEpisode()
            if (i + 1) % 100 == 0:
                mse = self.mseStateActionCompute(self.qMonteCarlo, self.q)
                mseList.append(mse)
                xList.append(i+1)
        return np.array(xList), np.array(mseList)
        
    def mseStepCurveFA(self, Lambda):
        self.Lambda = Lambda
        self.MonteCarloStatsInitialize()
        mseList = []
        xList = []
        for i in range(50000):
            self.faOneEpisode()
            if (i + 1) % 100 == 0:
                mse = self.mseStateActionComputeFA(self.qMonteCarlo, self.w)
                mseList.append(mse)
                xList.append(i+1)
        return np.array(xList), np.array(mseList)
                
            
           
            
    def mseStateActionCompute(self, q0, q1):
        mse = 0
        for state in list(q0.keys()):
            for action in list(q0[state].keys()):
                mse += np.square(q0[state][action] - q1[state][action])
        return mse
        
    def mseStateActionComputeFA(self, q0, w):
        mse = 0
        for state in list(q0):
            for action in list(q0[state].keys()):
                qMC = q0[state][action]
                qFA = self.valueCompute(state, action, w)
                mse += np.square(qMC - qFA)
        return mse

    def valueCompute(self, state, action, w):
        statePlayer, stateDealer = state.split('+')
        statePlayer, stateDealer = int(statePlayer), int(stateDealer)
        feature = self.featureEncode(stateDealer, statePlayer, action)
        return np.dot(w.T, feature)[0,0]

                
        
    def faTDLambdaControl(self, Lambda):
        self.alpha = 0.01
        self.epsilon = 0.05
        self.Lambda = Lambda
        self.faInitialize()
        for _ in range(self.faIterRound):
            self.faOneEpisode()
        
    
        
    def faInitialize(self):
        self.w = np.zeros([36, 1])
        
        
    
    def inRangeJudge(self, rangeList, x):
        if x >= rangeList[0] and x <= rangeList[1]:
            return True
        else:
            return False
            
    def featureEncode(self, stateDealer, statePlayer, action):
        feature = np.zeros([36, 1])
        for i in range(len(self.dealerSplit)):
            if self.inRangeJudge(self.dealerSplit[i], stateDealer):
                basei = 12 * i
            else:
                continue
            for j in range(len(self.playerSplit)):
                if self.inRangeJudge(self.playerSplit[j], statePlayer):
                    basej = 2 * j
                else:
                    continue
                for k in range(len(self.actionList)):
                    if self.actionList[k] == action:
                        basek = k
                        feature[basei + basej + basek, 0] = 1
        return feature
        
   

cl = cardRedBlack()
#cl.BackTDLambdaControl(0)
#cl.MonteCarloControl()
#cl.TDCompare()
cl.TDCompareFA()






