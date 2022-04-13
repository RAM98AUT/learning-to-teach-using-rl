"""This file fits a RL agent (or loads it), evaluates it against baselines
and plots the curricula for the skillbuilder dataset. A variety of models have been 
trained and can be loaded by setting the new_fit parameter to false
Which configurations concerning the other parameters have been trained can be seen 
in the naming of the models (in folder Saved Models - Skillbuilder)
Naming schema is as follows: Algorithm_timestepsPolicy_Environment_discountfactor_number of exercises
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
#Load skillbuilder parameter instead

params_file = 'output/torchbkt/block_params_final_fit.pkl'
with open(params_file, 'rb') as out: 
            block_obj = pickle.load(out)

block_params = {
    'l0': block_obj.l0_list,
    'transition': block_obj.transition_list,
    'slip': block_obj.slip_list,
    'guess': block_obj.guess_list,
    'blocks': block_obj.block_lengths,
    'n_skills': len(block_obj.skill_list)
}
max_skills = 4

# Number of evaluation episodes
n_eval_ep = 1
#Number of exercises
n_exercises = 300

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
    skill_path = os.path.join("RL","Saved Models - Skillbuilder",name)
    
    #Generate Exercise Types
    student_simulator = Simulator_v3(**block_params,max_skills=max_skills)
    exercises = student_simulator._gen_exercises()
    
    #Initialise Environment and check if working properly
    env = env_dict[env_number](n_exercises=n_exercises,
                   exercise_types=exercises,
                   **block_params,reward=[10,1,-1])
    check_env(env) 
    
    vec_env = DummyVecEnv([lambda: env])
     
    # Train/Load a model
    if new_fit:        
        model = used_model(policy,vec_env,verbose=1,gamma=discount_factor) # for PPO2 with LSTM include nminibatches=1
        model.learn(total_timesteps=timesteps)
        model.save(skill_path)
    else:
        model = used_model.load(skill_path,env=vec_env)
    
    
    
    # Evaluation
    reward_model = evaluate_policy(model,vec_env,n_eval_episodes=n_eval_ep,deterministic=True)
    
    eval_ob = Evaluation(env,max_skills,block_params["blocks"],env_number)
    eval_ob_2 = Evaluation(vec_env,max_skills,block_params["blocks"],env_number)
    reward_model_2,curr,info_2 = eval_ob_2.evaluate(model.predict,n_eval_ep,True)
    
    #other policies
    reward_rand,curr_rand,info_rand = eval_ob.evaluate("random",n_eval_ep,True)
    
    reward_greedy_single,curr_single,info_single = eval_ob.evaluate("greedy_single_pol",n_eval_ep,curriculum=True)
    reward_greedy_block,curr_block,info_block = eval_ob.evaluate("greedy_block_pol",n_eval_ep,curriculum=True)
    
    
    # Plot curriculum
    plot_curriculum(curr_rand,"Random policy",block_params["n_skills"],n_exercises,block_params["blocks"],max_skills,ticks=False,size=2)
    plot_curriculum(curr_single,"Single policy",block_params["n_skills"],n_exercises,block_params["blocks"],max_skills,ticks=False,size=2)
    plot_curriculum(curr_block,"Block policy",block_params["n_skills"],n_exercises,block_params["blocks"],max_skills,ticks=False,size=2)
    plot_curriculum(curr,"Learned policy",block_params["n_skills"],n_exercises,block_params["blocks"],max_skills,ticks=False,size=2)
      
    #print results
    print("Average model reward:",round(reward_model_2[0],2))
    print("Average reward when using a random policy:",round(reward_rand,2))
    print("Average reward when using a greedy single policy:",round(reward_greedy_single,2))
    print("Average reward when using a greedy block policy:",round(reward_greedy_block,2))
    
