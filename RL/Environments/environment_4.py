#Gym
import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

#Helpers
import numpy as np
import random
import os
import copy



class SkillEnv_4(Env):
    """ A class based on gym that provides the basis for Reinforcement learning
    The environment is initialised with necessary parameters and the action space, observation space and current state
    This environtment provides the agent with the number 
    of correct/incorrect responses so far as observation
    Reward: 
        Sum Probabilities of all skills
    
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
    reward: 
        not relevant for this environment
    
    """

    def __init__(self,blocks,n_skills,n_exercises,exercise_types,l0,transition,slip,guess,reward=None):
        """inits a environment with action space, observation space,current state, 
        number of skills and exercises and fitted BKT parameters"""
        
        self.transition = np.array(transition)
        self.slip = slip
        self.guess = guess
        self.n_exercises = n_exercises
        self.l0 = l0
        self.n_skills = n_skills
        self.exercise_types = exercise_types.astype(int)
        
        self.prob_state = np.array(self.l0, dtype=np.float32)
        self.responses = np.array([0,0])
        self.state = np.random.binomial(n=1,p=self.l0, size=self.n_skills)
        self.action_space = Discrete(exercise_types.shape[0]) 
        self.observation_space = MultiDiscrete([n_exercises,n_exercises])
        self.learn_length = self.n_exercises
    
    def step(self,action):
        
        """
        Arguments
        ------------
        action: int
                Provides the number of an exercise_type
        Output
        ------------
        responses: (n_skills,)
                correct/wrong responses of the stuedet till now
        reward: int
            Reward that a student gets for the action/exercises - sum of learning probabilities
        done: boolean
            Provides the info if the end of all operations has been reached
        info: str
            can contain additional information
        """
        
        #extract skills from action/exercise type
        skills = np.where(self.exercise_types[action,:-1]==1)[0]
        block_id = self.exercise_types[action,-1]
        
        # Slip/Guess for exercise
        slip = self.slip[block_id]
        guess = self.guess[block_id]
        
        
        #Exercise
        learned = np.min(self.state[skills])
        probs = learned*(1-slip) + (1-learned)*guess
        evidence = np.random.binomial(n=1, p=probs)
        
        # add to responses
        self.responses[evidence]+=1
        
        
        state_for_ex = self.prob_state[skills]
        
        #update learning probabilities
        new_states = np.zeros_like(state_for_ex)
        
                
        #Calculate #P(sk learned|evidence) for all relevant skills 
        #as estimator for the updated learning probability we use P(sk learned|evidence) + (1-P(sk learned|evidence))*transition
        if evidence == 0:
            for i in range(skills.shape[0]):
                #P(sk learned|wrong)=P(wrong|sk learned)P(sk learned)/(... + P(wrong|sk not learned)*P(sk not learned))
                prob = state_for_ex[i]
                prob_rest_skills = np.prod(np.delete(state_for_ex,i)) if skills.shape[0]>1 else 1
                prob_wrong_learned = prob_rest_skills * slip + (1-prob_rest_skills)*(1-guess)
                prob_under_ev = (prob_wrong_learned * prob) / (prob_wrong_learned * prob + (1 - guess) *(1 - prob))
                new_states[i] = prob_under_ev + (1-prob_under_ev) * self.transition[skills[i]]
        else:
            for i in range(skills.shape[0]):
                #P(sk learned|correct)=P(correct|sk learned)P(sk learned)/(... + P(corr|sk not learned)*P(sk not learned))
                prob = state_for_ex[i]
                prob_rest_skills = np.prod(np.delete(state_for_ex,i)) if skills.shape[0]>1 else 1
                prob_correct_learned = prob_rest_skills * (1-slip) + (1-prob_rest_skills)*guess
                prob_under_ev = (prob_correct_learned * prob ) / (prob_correct_learned * prob + guess *(1 - prob))
                new_states[i] = prob_under_ev + (1-prob_under_ev) * self.transition[skills[i]]
        
        #update states
        self.prob_state[skills] = new_states
        
        
        #Calculate reward
        reward = float(sum(self.prob_state))
        
        #Change state with action
        transi = np.random.binomial(n=1, p=self.transition[skills])
        self.state[skills] = np.maximum(transi,self.state[skills])
        
        
        #learn length (number of exercises that the student will still solve)
        self.learn_length -= 1
        if self.learn_length<=0:
            done = True
        else:
            done = False
        
        info ={"state": self.state,"last_ex":evidence}
        return self.responses, reward, done, info
    
    def render(self):
        pass
    
    def reset(self):
        """
        Resets state and learn length

        Returns
        -------
         responses: (2,)
                incorrectly/correctly answered questions in current episode
        """
        self.prob_state = np.array(self.l0, dtype=np.float32)
        self.responses = np.array([0,0])
        self.state = np.random.binomial(n=1,p=self.l0, size=self.n_skills)
        self.learn_length = self.n_exercises
        return self.responses


    
    