# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 18:42:24 2016

@author: Kireeti
"""
import numpy as np
class LearningAgent:
    currentState= None
    actions=[]
    E_cap=0.0
    P_cap=0.0
    epsilon=0.01
    E_min=0
    E_init=0
    kLookAhead=2
    
    def __init__(self, currentState, actions, E_cap,P_cap,epsilon):
        self.currentState=currentState
        self.actions=actions
        self.E_cap=E_cap
        self.P_cap=P_cap          
        self.epsilon=epsilon
        self.E_min = (1-0.80)*self.E_cap #depth of discharge
        self.E_init = (0.30)*self.E_cap #30% of charge every morning
        


    def getLegalActions(self, currentState, environment):
            '''
            Calculate and return allowable action set
            Output: List of indices of allowable actions
            '''
            energy_level = currentState[1] #1->energy level
            lower_bound = max(self.E_min - energy_level, -self.P_cap)
            upper_bound = min(self.E_max - energy_level, self.P_cap)
            max_bin = int(np.digitize(upper_bound, self.actions, right=True)) ###
            min_bin = int(np.digitize(lower_bound, self.actions, right=True)) ###            
            legal_actions = []
            for k in range(min_bin, max_bin+1):
                legal_actions.append(k)
            return legal_actions
    
    def getAction(self, state, model, environment, k, gama):
         """returns a tupple of optimal actions , reward."""
         if k==0:
             legalActions=self.getLegalActions(self.currentState, environment)
             flag=0             
             QValue=None
             optimalAction=None
             for action in legalActions:
                 if flag==0:
                     QValue=model.predictQvalue(state,action)
                     optimalAction=action
                     flag=1
                 else:
                     if QValue < model.predictQvalue(state,action):
                        QValue=model.predictQvalue(state,action)
                         optimalAction=action
             return [optimalAction],gama*QValue
         else:
             legalActions=self.getLegalActions(self.currentState,environment)
             optimalReward=None
             optimalActions=[]
             currentOPtimalAction=None
             for action in legalActions:
                 #check the return order for getNextState here
                 nextState,current_reward, isValid =self.getNextState(model,environment,k-1)
                 if isValid:
                     continue                    
                 actionTupples,reward=self.getAction(nextState,model,environment,k-1,gama)
                 if optimalReward==None:
                     continue
                 if optimalReward < current_reward + gama*reward:
                     optimalReward = current_reward + gama*reward
                     currentOPtimalAction = action
                     optimalActions=actionTupples
             return currentOPtimalAction+optimalActions, optimalReward
                    
            
                 
                 
                 
                

        
    
        