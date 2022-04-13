#import Environments
import numpy as np


class Evaluation:
    
    "Class for evaluating baselines"
    """
    

    Parameters
    ----------
    env : object of gym class environment
    max_skills : int
        max_skills per exercise
    blocks : list
        describes blockstructure
    env_number : int
        describes which kind of envirnoment is used
    mastery_threshold : float, optional
         Describes when a skill with probabilistic learning state counts as mastered
         The default is 0.95.

    """
    
    def __init__(self,env,max_skills,blocks,env_number,mastery_threshold = 0.95):
        "Inits an object of class Evaluation"
        self.env = env
        self.env_number = env_number
        self.max_skills = max_skills
        self.blocks = blocks
        self.mastery_threshold = mastery_threshold
        
    def evaluate(self,action_func,eps,curriculum=False):
        """
        Performs the evaluation given an environment, a class and an algorithm
        Parameters
        ----------
        action_func : function
            Defines which policy should be used for evaluation
        eps : int
            Number of episodes that are run through
        curriculum : boolean, optional
            defines if a curriculum should also be shown & saved. The default is False.
    
        Returns
        -------
        med_reward : int
            Average reward the policy achieved
        
        curr: list
            list of skills that were queried in the last episode
        other: list
            consists of further interesting information (evaluation array, learned skills, std)
        
    
        """
        eval_arr = []
        skill_arr = []
        for episode in range(eps):
            obs = self.env.reset()         
            done = False
            score = 0
            if curriculum and episode == eps-1:
                if isinstance(action_func,str):
                    curr = np.full((self.env.learn_length,self.max_skills),None)
                else:
                    curr = np.full((self.env.get_attr("learn_length")[0],self.max_skills),None) 
                i = 0
            while not done:
                if action_func=="random":
                    action = self.random(obs) 
                elif action_func=="greedy_single_pol":
                    action = self.greedy_single_pol(obs) 
                elif action_func=="greedy_block_pol":
                    action = self.greedy_block_pol(obs) 
                else:
                    action,_ = action_func(obs)
                obs, reward, done, info = self.env.step(action)
                if curriculum and episode == eps-1:
                    if isinstance(action_func,str):
                        skills = np.where(self.env.exercise_types[action,:-1])[0]
                    else:
                        skills = np.where(self.env.get_attr("exercise_types")[0][action[0],:-1])[0]
                    curr[i,:len(skills)]=skills
                    i += 1
                score += reward
            eval_arr.append(score)
            skill_arr.append(sum(self.env.state if isinstance(action_func,str) else info[0]["state"]))
        med_skill = sum(skill_arr)/len(skill_arr)
        med_reward=sum(eval_arr)/len(eval_arr)
        if curriculum:
            return med_reward, curr,[eval_arr,np.std(eval_arr),med_skill]
        else:
            return med_reward, curr,[eval_arr,np.std(eval_arr),med_skill]
        

    def random(self,obs):
        """
        Chooses a random possible action.
       

        Parameters
        ----------
        obs: list (n_skills)
            current observation
            
        Returns
        -------
        result : int
            index of the exercisetype that should be performed (=action)


        """
        return self.env.action_space.sample()
        


    def greedy_single_pol(self,obs):
        """
        Provides as output the endcoded action for giving a student the action 
        that includes only his "lowest" (namingwise) not learned skill
        (depending if the function gets the probabilistic or true learning states learned
         means over the mastery threshold or =1)
        eg obs = [0,0,0] -> student will get an exercise that includes only skill 0
        if all skills are learned a random action is chosen

        Parameters
        ----------
        obs: list (n_skills)
            current observation

        Returns
        -------
        result : int
            index of the exercisetype that should be performed (=action)

        """
        if self.env_number in [4,6]:
            obs = self.env.prob_state
        ex_array = self.env.exercise_types
        if not np.all(obs>self.mastery_threshold):
            first_one = np.where(obs<self.mastery_threshold)[0][0]
            exercise = [0]*(ex_array.shape[1]-1)
            exercise[first_one] = 1
            result = ex_array[:,:-1].tolist().index(exercise)
        else:
            result = np.random.randint(ex_array.shape[0])
        return result

    def greedy_block_pol(self,obs):
        """
        Provides as output the endcoded action for giving a student the action 
        that includes as much skills as possible of the block of his  "lowest" (namingwise) not learned skill
        (depending if the function gets the probabilistic or true learning states learned
         means over the mastery threshold or =1)
        eg obs =[0,1,1,0,0,0] with 2 blocks [3,3],max skills=2-> 
        student will get an exercise that includes two skills from block 0 (the lowest unlearned, rest random)
        so exercise_type [1,1,0,0,0,0]
        if all skills are learned a random action is chosen


        Parameters
        ----------
        obs: list (n_skills)
            current observation

        Returns
        -------
        result : int
            index of the exercisetype that should be performed (=action)

        """
     
        
        if self.env_number in [4,6]:
            obs = self.env.prob_state
        ex_array = self.env.exercise_types
        if not np.all(obs>self.mastery_threshold):
            first_one = np.where(obs<self.mastery_threshold)[0][0]
            skill_count = 0
            block = None
            for i in range(len(self.blocks)): #find block to first_one skill
                skill_count+=self.blocks[i]
                if first_one<skill_count:
                    block = i
                    break
            possible_exercises = ex_array[(ex_array[:,-1]==block)
                                          & (ex_array[:,first_one]==1) 
                                          & (np.sum(ex_array[:,:-1],axis=1)==min(self.max_skills,self.blocks[i]))]
            exercise = possible_exercises[np.random.choice(possible_exercises.shape[0],size=1),:]
            result = ex_array.tolist().index(exercise.tolist()[0])
        else:
            result = np.random.randint(ex_array.shape[0])
        return result





    