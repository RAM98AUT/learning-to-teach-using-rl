"""This file fits a RL agent (or loads it), evaluates it against baselines
and plots the curricula for a simulated data set. The evaluation can be done 
on a environment with different parameters (to test on simulated "true" data)
Always results for a fast and a slow group of students are omitted

A variety of models have been trained and can be loaded by setting the new_fit parameter to false
All models trained with this file have been trained with the included block_params/maxskills.
Which configurations concerning the other parameters have been trained can be seen 
in the naming of the models (in folder Saved Models - True Evaluation)
Naming schema is as follows: Algorithm_timestepsPolicy_Environment_discountfactor_number of exercises

Necessary data can be generated with Blockbkt_2groups.
"""
from RL import *
from simulator import Simulator_v3
import os
import numpy as np
import pickle
import random

# Stable baseline
from stable_baselines import A2C,DQN,PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.env_checker import check_env

#plotting
import matplotlib.pyplot as plt


####### Input parameters ##########
# Parameters for BKT that are used in the environment

with open('output/torchbkt/block_params_simulated_fitted.pkl', 'rb') as out:
     fitted_params = pickle.load(out)

with open('output/torchbkt/block_params_simulated_slow.pkl', 'rb') as out:
     true_params_slow = pickle.load(out)

with open('output/torchbkt/block_params_simulated_fast.pkl', 'rb') as out:
     true_params_fast = pickle.load(out)
     
max_skills = 3

# Number of evaluation episodes
n_eval_ep = 100
#Number of exercises
n_exercises = 50

# Fitting or loading an old model
new_fit = False

#Which agent/poliy with how many timesteps, which discountfactor
policy = "MlpPolicy"
used_model = A2C #for PPO2 with LSTM include nminibatches=1 in the code when initialising the model
timesteps = 1000000
discount_factor = 0.90 #default 0.99 if not stated otherwise

#Which environment
env_number = 5


##############

if __name__ == '__main__':
    
    # Right environment
    env_dict = {1: SkillEnv_1, 2: SkillEnv_2, 3: SkillEnv_3, 4: SkillEnv_4, 5: SkillEnv_5,6: SkillEnv_6}
    
    #path for saving model
    name = "_".join([used_model.__name__,str(timesteps)+policy,env_dict[env_number].__name__,"df"+str(discount_factor*100)[:2],str(n_exercises)])
    skill_path = os.path.join("RL","Saved Models - True Evaluation",name)
    
    #Generate Exercise Types
    student_simulator = Simulator_v3(**fitted_params,max_skills=max_skills)
    exercises = student_simulator._gen_exercises()
    
    #Initialise Environment and check if working properly
    env = env_dict[env_number](n_exercises=n_exercises,
                   exercise_types=exercises,
                   **fitted_params,reward=[10,1,-1])
    check_env(env) 
    
    vec_env = DummyVecEnv([lambda: env])
     
    # Train/Load a model
    if new_fit:        
        model = used_model(policy,vec_env,verbose=1,gamma=discount_factor) # for PPO2 with LSTM include nminibatches=1
        model.learn(total_timesteps=timesteps)
        model.save(skill_path)
    else:
        model = used_model.load(skill_path,env=vec_env)
    
    
    for params,name in zip([true_params_fast,true_params_slow,fitted_params],["fast","slow","fitted"]):
        # Evaluation fast
        #Initialise Evaluation Environment and check if working properly
        eval_env = env_dict[env_number](n_exercises=n_exercises,
                       exercise_types=exercises,
                       **params,reward=[10,1,-1])
        check_env(eval_env) 
        
        eval_vec_env = DummyVecEnv([lambda: eval_env])
        
        #Integrated reward evaluation: Performs always worse and can not get all wanted information
        #reward_model = evaluate_policy(model,eval_vec_env,n_eval_episodes=n_eval_ep,deterministic=True)
        
        eval_ob = Evaluation(eval_env,max_skills,fitted_params["blocks"],env_number)
        eval_ob_2 = Evaluation(eval_vec_env,max_skills,fitted_params["blocks"],env_number)
        reward_model_2,curr,info_2 = eval_ob_2.evaluate(model.predict,n_eval_ep,True)
        
        #other policies
        reward_rand,curr_rand,info_rand = eval_ob.evaluate("random",n_eval_ep,True)
        reward_greedy_single,curr_single,info_single = eval_ob.evaluate("greedy_single_pol",n_eval_ep,curriculum=True)
        reward_greedy_block,curr_block,info_block = eval_ob.evaluate("greedy_block_pol",n_eval_ep,curriculum=True)
        
        
        # Plot curriculum
        plot_curriculum(curr_rand,"Random policy",fitted_params["n_skills"],n_exercises,fitted_params["blocks"],max_skills)
        plot_curriculum(curr_single,"Single policy",fitted_params["n_skills"],n_exercises,fitted_params["blocks"],max_skills)
        plot_curriculum(curr_block,"Block policy",fitted_params["n_skills"],n_exercises,fitted_params["blocks"],max_skills)
        plot_curriculum(curr,"Learned policy",fitted_params["n_skills"],n_exercises,fitted_params["blocks"],max_skills)
          
        #print results
        print("Average model reward using",name,"params:",round(reward_model_2[0],2),", skills_learned:",round(info_2[-1],1))
        print("Average reward when using a random policy with",name,"params:",round(reward_rand,2),", skills_learned:",round(info_rand[-1],1))
        print("Average reward when using a greedy single policy with",name,"params:",round(reward_greedy_single,2),", skills_learned:",round(info_single[-1],1))
        print("Average reward when using a greedy block policy with",name,"params:",round(reward_greedy_block,2),", skills_learned:",round(info_block[-1],1))
    
  