# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 18:42:24 2016

@author: Kireeti
"""
import numpy as np
import sys
import random

class LearningAgent:
    
    def __init__(self, currentState, actions, E_cap,P_cap,epsilon):
        self.debug=True
        self.currentState=currentState
        self.actions=actions
        self.E_max=E_cap
        self.P_cap=P_cap          
        self.epsilon=epsilon
        self.E_min = (1-0.80)*self.E_max #depth of discharge
        self.E_init = (0.30)*self.E_max #30% of charge every morning
        


    def getLegalActions(self, currentState):
            '''
            Calculate and return allowable action set
            Output: List of indices of allowable actions
            '''
            energy_level = currentState[1] # 1.8
            lower_bound = max(self.E_min - energy_level, -self.P_cap) # 3.0
            upper_bound = min(self.E_max - energy_level, self.P_cap) # -0.6
            
            max_bin = int(np.digitize(upper_bound, self.actions, right=True)) ###
            min_bin = int(np.digitize(lower_bound, self.actions, right=True)) ###            
            
            legal_actions = []
            for k in range(min_bin, max_bin):
                legal_actions.append(k)
            '''
            if self.debug:
                file=open('debug.txt','a')
                file.write('in getLegalActions in LearningAgent.py.......\n')
                file.write("currentState = " + str(currentState) + " upper_bound = " + str(upper_bound) + "lower_bound = " + str(lower_bound) + " max_bin, min_bin = " + str((max_bin, min_bin)) + "\n") 
                file.close()
            '''
            return legal_actions

    def exploration(self, episode_number, time_step, environment, k):
            legalActions = self.getLegalActions(self.currentState)   
            action_index = random.choice(legalActions)
            state, cumulativeReward = environment.getCumulativeReward(episode_number, time_step, k, [self.actions[action_index]])
            #print state, cumulativeReward
            return [action_index], cumulativeReward
    
    def getAction(self,episodeNumber, state, model, environment, k, gama,timeStamp):
         """returns a tupple of optimal actions , reward."""
         if self.getLegalActions(state)==None:
                return None,-100000

         if k==0:
             legalActions=self.getLegalActions(state)  #check this
             '''
             if self.debug and len(legalActions)== 0:
                file=open('debug.txt','a')
                file.write('legalActions should not be empty in getAction in LearningAgent.py.......\n')
                file.write(" episodeNumber = " + " " + str(episodeNumber) + " state = " + str(state) + " k =" + str(k) + " timeStamp = " + timeStamp + "\n") 
                file.close()
            '''
             flag=0             
             Qvalues =[(legalActions[i], model.predictQvalue(state,self,[j])) for i,j in enumerate(legalActions)]
             QValue = Qvalues[0][1]
             optimalAction = Qvalues[0][0]

             for a,q in Qvalues:
                if q >= QValue:
                    optimalAction = a
                    QValue = q 
             return [optimalAction], gama*QValue
         else:
             legalActions = self.getLegalActions(state)  #check this                
             optimalReward = None
             optimalActions = []
             currentOPtimalAction = None

             for action in legalActions:
                 #check the return order for getNextState here
                 #print 'getAction: ',self.actions, action
                 nextState, current_reward, isValid = environment.getNextState(episodeNumber,timeStamp, state, self.actions[action]) 
                 
                 if isValid:
                     continue                    
                 actionTupples, reward = self.getAction(episodeNumber,nextState, model, environment, k-1, gama, timeStamp+1)


                 
                 if optimalReward < current_reward + gama*reward:
                     optimalReward = current_reward + gama*reward
                     currentOPtimalAction = action
                     optimalActions = actionTupples
             
             return [currentOPtimalAction] + optimalActions, optimalReward

                    
            
                 
                 
                 
                

        
    
        