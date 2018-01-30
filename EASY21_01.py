# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:07:43 2018

@author: admin
"""

import numpy as np


class cardRedBlack(object):
    def __init__(self):
        self.MonteCarloIterRound = 1000000
        self.MonteCarloN0 = 100
        self.actionList = ['hit', 'stick']
    
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
            self.stateVisits[str(statePlayer) + '+' + str(stateDealer)] += 1
            epsilon = self.MonteCarloN0 * 1.0 / (self.MonteCarloN0 + self.stateVisits[str(statePlayer) + '+' + str(stateDealer)])
            action = self.MonteCarloTakeAction(epsilon, statePlayer, stateDealer)
            self.stateactionVisits[str(statePlayer) + '+' + str(stateDealer)][action] += 1
            pairStateAction.append([str(statePlayer) + '+' + str(stateDealer), action])
            termination, reward = self.step(action)
        for pair in pairStateAction:
            self.q[pair[0]][pair[1]] += 1.0 / self.stateactionVisits[pair[0]][pair[1]] * reward
        self.rewardPrint += reward
            
            
            
    def MonteCarloTakeAction(self, epsilon, statePlayer, stateDealer):
        rand0 = np.random.rand(1)[0]
        if rand0 < epsilon:
            rand1 = np.random.randint(2)
            action = self.actionList[rand1]
        else:
            if self.q[str(statePlayer) + '+' + str(stateDealer)]['hit'] > self.q[str(statePlayer) + '+' + str(stateDealer)]['stick']:
                action = 'hit'
            elif self.q[str(statePlayer) + '+' + str(stateDealer)]['hit'] < self.q[str(statePlayer) + '+' + str(stateDealer)]['stick']:
                action = 'stick'
            else:
                rand1 = np.random.randint(2)
                action = self.actionList[rand1]
        return action   
        
    def MonteCarloResultPlot(self):
        xList, yList, zList = [], [], []
        for i in range(1, 11):
            for j in range(1, 22):
                

    

cl = cardRedBlack()
cl.MonteCarloControl()






















