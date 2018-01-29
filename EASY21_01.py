# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:07:43 2018

@author: admin
"""

import numpy as np


class cardRedBlack(object):
    def __init__(self):
        pass
    
    def stateInitialize(self):
        self.state = dict()
        self.state['dealer'] = np.random.randint(1, 11)
        self.state['player'] = np.random.randint(1, 11)
        return 'unfinished',0
        
    def step(self, action):
        if action == 'hit':
            self.state['player'] += self.cardpoolSimulator()
            if self.state['player'] < 0 or self.state['player'] > 21:
                return 'finished', -1
            else:
                return 'unfinished', 0
        else:
            dealerTurn = True
            while dealerTurn == True:
                self.state['dealer'] += self.cardpoolSimulator()
                if self.state['dealer'] < 0 or self.state['dealer'] > 21:
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
        























