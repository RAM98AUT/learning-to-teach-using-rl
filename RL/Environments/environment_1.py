#Gym
import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

#Helpers
import numpy as np
import random
import os



# Environment

class SkillEnv_1(Env):
    """ A class based on gym that provides the basis for Reinforcement learning
    The environment is initialised with necessary parameters and the action space, observation space and current state
    

    This environtment knows all learning states of the current skills
    Reward: parameter given by reward array  
        reward[0] if new skill is learned
        reward[1] if the exercise includes unlearned skills
        reward[2] if the exercise only includes already learned skills
    
    Attributes
    ------------
    blocks : (n_blocks)
        List that represents how many skills each block should contain
    n_exercises: int
        number of exercises
    n_skills: int
        number of skills
    exercise_types: (n_exercises,max_skills)
        array that represents the different exercise types according to max skills, skill number and blocks
    l0 : (n_skills,) array-like or float
        The priors for the skills. Can be either float for a common prior.
        Or array-like of length n_skills for skill-individual priors.
    transition : (n_skills,) array-like
        The transition probabilities for the skills.
    slip : (n_blocks,) array-like
        The slip probabilities for the blocks.
    guess : (n_blocks,) array-like
        The guess probabilities for the blocks.
    reward: (3,)
        Defines reward for different scenarios
    
    """
    
    def __init__(self,blocks,n_skills,n_exercises,exercise_types,l0,transition,slip,guess,reward=[10,1,-1]):
        """inits a environment with action space, observation space,current state, 
        number of skills and exercises and fitted BKT parameters"""
        
        self.transition = np.array(transition)
        self.slip = slip
        self.guess = guess
        self.n_exercises = n_exercises
        self.l0 = l0
        self.n_skills = n_skills
        self.exercise_types = exercise_types.astype(int)
        
        self.state = np.random.binomial(n=1,p=self.l0, size=self.n_skills) 
        self.action_space = Discrete(exercise_types.shape[0]) 
        self.observation_space = MultiBinary(n_skills)
        self.learn_length = self.n_exercises
        self.reward=reward
    
    def step(self,action):
        
        """
        Arguments
        ------------
        action: int
                Provides the number of an exercise_type
        Output
        ------------
        state: (n_skills,)
                current learning state of a student for each skill
        reward: int
            Reward that a student gets for the action/exercises
        done: boolean
            Provides the info if the end of all operations has been reached
        info: str
            can contain additional information
        """
        skills = np.where(self.exercise_types[action,:-1]==1)[0]
        block_id = self.exercise_types[action,-1]
        
        know_all = np.all(self.state[skills])
        know_all_skills = np.all(self.state)
        
        
        #Exercise
        state_for_ex = np.min(self.state[skills])
        probs = state_for_ex*(1-self.slip[block_id]) + (1-state_for_ex)*(self.guess[block_id])
        evidence = np.random.binomial(n=1, p=probs)
        
        #Change state with action
        transi = np.random.binomial(n=1, p=self.transition[skills])
        if sum(transi-np.minimum(transi,self.state[skills]))>0:
            new_learned = True
        else:
            new_learned = False
        self.state[skills] = np.maximum(transi,self.state[skills]) 
        
        #Decrease learn length (number of exercises that the student will still solve)
        self.learn_length -= 1
        
        #Calculate reward
                
        if know_all and not know_all_skills:
            reward = self.reward[2]
        elif new_learned:
            reward = self.reward[0]
        elif know_all and know_all_skills:
            reward = self.reward[1] 
        else:
            reward = self.reward[1]
        
        
        if self.learn_length<=0:
            done = True
        else:
            done = False
        
        info ={"state": self.state,"last_ex":evidence}
        return self.state, reward, done, info
    
    def render(self):
        pass
    
    def reset(self):
        """
        Resets state and learn length

        Returns
        -------
         state: (n_skills,)
                current learning state of a student for each skill

        """
        self.learn_length = self.n_exercises
        self.state = np.random.binomial(n=1,p=self.l0, size=self.n_skills)
        return self.state
    




    
    