import numpy as np
import pandas as pd
import itertools
from scipy.special import comb



class Simulator_v3:
    """
    A class to simulate students with a common performance from a BKT model.

    When simulating more than one skill per exercise, the following assumptions hold:
        - If all affected skills are in the learned state, there is still the chance to slip.
        - If any affected skill is in the unlearned state, the correct answer has to be guessed.
    n_skills represents the number of skills
    n_blocks represents the number of blocks
    It can be specified how many skills should be included in each block
    Exercises can include up to max_skills skills (but also less)

    Attributes
    ----------
    blocks : (n_blocks)
        List that represents how many skills each block should contain
    l0 : (n_skills,) array-like or float
        The priors for the skills. Can be either float for a common prior.
        Or array-like of length n_skills for skill-individual priors.
    transition : (n_skills,) array-like
        The transition probabilities for the skills.
    slip : (n_blocks,) array-like
        The slip probabilities for the blocks.
    guess : (n_blocks,) array-like
        The guess probabilities for the blocks.
    min_id : int
        The user_id of the first student to be sampled.
        The consecutive students have user_id min_id+1, min_id+2, ...
    max_skills : int
    The maximum number of skills a single exercise can comprise. (default: 1)
    """
    
    def __init__(self, blocks,n_skills, l0, transition, slip, guess, min_id=0,max_skills=1):
        """Inits a simulator with args blocks,n_skills l0, transition, slip, guess, min_id and max_skills.
        It also checks all sizes as using wrong sizes leads to a non working simulator"""
        self.blocks = blocks
        self.n_skills = n_skills
        self.n_blocks = len(blocks)
        self.l0 = l0
        self.transition = np.array(transition)
        self.slip = slip
        self.guess = guess
        self.min_id = min_id
        self.max_skills = max_skills
        
        #Errors
        if self.n_skills != sum(self.blocks):
            raise ValueError("Number of skills n_skills does not agree with skillnumber specified in blocks")
        if len(self.slip) != self.n_blocks or len(self.guess) != self.n_blocks:
            raise ValueError("Size of slip/guess tuple does not agree with the number of blocks")
        if self.transition.shape[0] != self.n_skills:
            raise ValueError("Size of transition array does not agree with the number of skills")
        
    def sample_students(self, n_students, n_exercises):
        """Samples sequences for multiple students.
        Arguments
        ---------
        n_students : int
            The number of students to be sampled.
        n_exercises : int
            n_exercises are provided to each student in every block,
            has to be divisible by number of skills
            The number of skills per block is proportional to the number of its skills
            eg n_skills=5, n_exercises=100, student gets 20 skills of a block with 2 exercises 
        """
        if not n_exercises % self.n_skills==0: 
            raise ValueError("Number of exercises has to be divisible by number of skills")
        # Generate exercises
        self.sim_ex = self._gen_exercises()
        # simulation
        students = []
        for student_id in range(self.min_id, self.min_id + n_students):
            new_student = self._sample_student(n_exercises, student_id)
            students.append(new_student)
        students = pd.concat(students).reset_index().rename(columns={'index': 'order_id'}) 
        return students
        
    def _sample_student(self, n_exercises, student_id):
        """Simulates a sequence for a single student.

        Arguments
        ---------
        n_exercises : int
            The number of exercises the student completes.
            
        student_id : int
            The identifier of the student.

        Returns
        -------
        DataFrame 
            Equivalent to the output of the _create_data_frame method.
        """
        # sample initial L and S matrix
        ex_per_skill = int(n_exercises/self.n_skills)
        n_exercises_p_block= [x*ex_per_skill for x in self.blocks]
        L = self._init_L(n_exercises)
        S = self._ex_for_student(n_exercises_p_block).astype("int")
        # update L matrix
        for i in range(n_exercises-1):
            L[i+1, :] = L[i, :]
            skill = np.where(S[i,:-1]==1)
            transi = np.random.binomial(n=1, p=self.transition[skill])
            L[i+1, skill] = np.maximum(transi, L[i, skill])
        # sample C matrix
        rel_L = L >= S[:, :-1]
        rel_L = np.min(rel_L, axis=1) # is any relevant skill unlearned
        slip_mask = np.repeat(self.slip, n_exercises_p_block)
        guess_mask = np.repeat(self.guess, n_exercises_p_block)
        probs = rel_L*(1-slip_mask) + (1-rel_L)*guess_mask
        C = np.random.binomial(n=1, p=probs)
        return self._create_data_frame(L, S, C, student_id)
        
    def _init_L(self, n_exercises):
        """Inits the matrix containing all latent state information.
        
        Arguments
        ---------
        n_exercises : int
            The number of exercises the sampled student completes.

        Returns
        -------
        (n_exercises, n_skills) ndarray
            A matrix where each row represents a timestep and each column represents a skill.
            The matrix is supposed to contain the hidden state information for each skill at each point in time.
        """
        L = np.zeros((n_exercises, self.n_skills))
        L[0, :] = np.random.binomial(n=1, p=self.l0, size=self.n_skills)
        return L
    
    def _create_data_frame(self, L, S, C, student_id):
        """Creates the output dataframe for a single sampled student.

        Arguments
        ---------
        L : (n_exercises, n_skills) ndarray
            The matrix containing the latent state for each timestep and skill.
        S : (n_exercises,) ndarray
            The array showing which skill was considered in each exercise.
        C : (n_exercises,) ndarray
            The array showing whether each exercise was solved correctly.
        student_id : int
            The identifier of the student.
        max_skills : int
            The maximum number of skills that can appear in a single exercise.
        Returns
        -------
        DataFrame 
            Basically a horizontal concatenation of the arguments.
            Columns: [Sk0, Sk1, Sk2, ..., skill_name0, skill_name1, ..., correct, user_id]         
        """
        data_frame = np.c_[L, S, C]
        columns = ['Sk' + str(i) for i in range(self.n_skills)] + ["skill_name" + str(i) for i in range(self.n_skills)] + ["block_id"]+ ['correct']
        data_frame = pd.DataFrame(data_frame, columns=columns).astype('int')
        data_frame['user_id'] = student_id
        return data_frame
    
    def _ex_for_student(self,n_exercises_p_block):
        """Selects exercises that the student has to solve.
        
        Arguments
        ---------
        n_exercises_p_block : (n_blocks)
            The number of exercises the student completes per block

        Returns
        -------
        (n_exercises*n_skills, n_skills+1) ndarray
            A matrix where each row represents a exercise and each column  if a skill is present or not. 
            The last column represents a block id.
        """       
        #with numpy
        student_exercises = np.zeros((0, self.n_skills+1), dtype=int)
        for num,i in enumerate(self.blocks):
            possible_exercises = self.sim_ex[np.where(self.sim_ex[:,-1] == num)]
            num_ex = possible_exercises.shape[0]
            ind = np.random.choice(num_ex, size=n_exercises_p_block[num], replace=True)
            student_exercises = np.append(student_exercises, possible_exercises[ind,:], 0)
        return student_exercises
    
    def _gen_exercises(self):
        """Creates all possible exercises according to the provided blockstructure.
        
        Arguments
        ---------

        Returns
        -------
        (possible exercise combinations * n_skills, n_skills+1) ndarray
            A matrix where each row represents a exercise and each column  if a skill is present or not. 
            The last column represents a block id
        """
        exercises = np.zeros((0, self.n_skills+1), dtype=int)
        for i in range(0, self.n_blocks):
            skills = range(sum(self.blocks[:i]), sum(self.blocks[:i+1]))
            for L in range(0, self.max_skills):
                temp_array = np.c_[np.zeros((int(comb(len(skills), L+1)), self.n_skills)), i*np.ones(int(comb(len(skills), L+1)))]
                for n,subset in enumerate(itertools.combinations(skills, L+1)):
                    temp_array[n,subset] = 1
                exercises = np.append(exercises,temp_array, 0)             
        return exercises
        