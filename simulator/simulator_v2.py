import numpy as np
import pandas as pd



class Simulator_v2:
    """
    A class to simulate students with a common performance from a BKT model.
    Each block includes n_skills/n_blocks skills
    Eeach skill can have their own prior,
    each block has own slip,guess and transition

    When simulating more than one skill per exercise, the following assumptions hold:
        - If all affected skills are in the learned state, there is still the chance to slip.
        - If any affected skill is in the unlearned state, the correct answer has to be guessed.

    Attributes
    ----------
    n_skills : int
        Number of different skills to be considered.
        Should be divisible by n_blocks.
    n_blocks : int
        The number of skill blocks from which exercises can have multiple skills.
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
    """
    
    def __init__(self, n_skills, n_blocks, l0, transition, slip, guess, min_id=0):
        """Inits a simulator with args n_skills, n_blocks, l0, transition, slip, guess and min_id."""
        assert(n_skills%n_blocks == 0)
        self.n_skills = n_skills
        self.n_blocks = n_blocks
        self.l0 = l0
        self.transition = np.array(transition)
        self.slip = slip
        self.guess = guess
        self.min_id = min_id
        
    def sample_students(self, n_students, n_exercises, choose_skill="_random_single", max_skills=1):
        """Samples sequences for multiple students.
        
        Arguments
        ---------
        n_students : int
            The number of students to be sampled.
        n_exercises : int
            The sequence length to be sampled for each student.
            Must be divisable by n_blocks.
        choose_skill : str
            The skill sampling scheme to be used. 
            One of {'_random_single', '_blockwise'}. (default: '_random_single')
        max_skills : int
            The maximum number of skills a single exercise can comprise. (default: 1)
            Has to be 1 for choose_skill='_random_single'.
            
        Returns
        -------
        students: DataFrame  (n_students*n_exercises,n_skills+max_skills+3)
            DataFrame containing all sampled information with n_students*n_exercises rows.
            Basically the vertical concatenation of multiple outputs from the _sample_student method.
        
        """
        # assertions
        assert(n_exercises%self.n_blocks == 0)
        if choose_skill=='_random_single': assert(max_skills==1)
        # simulation
        students = []
        for student_id in range(self.min_id, self.min_id + n_students):
            new_student = self._sample_student(n_exercises, student_id, choose_skill, max_skills)
            students.append(new_student)
        students = pd.concat(students).reset_index().rename(columns={'index': 'order_id'}) 
        return students
        
    def _sample_student(self, n_exercises, student_id, choose_skill, max_skills):
        """Simulates a sequence for a single student.

        Arguments
        ---------
        n_exercises : int
            The number of exercises the student completes.
            Must be divisable by n_blocks.
        student_id : int
            The identifier of the student.

        Returns
        -------
        DataFrame 
            Equivalent to the output of the _create_data_frame method.
        """
        # sample initial L and S matrix
        L = self._init_L(n_exercises)
        skill_func = getattr(self, choose_skill)
        S = skill_func(n_exercises, max_skills)
        # update L matrix
        for i in range(n_exercises-1):
            L[i+1, :] = L[i, :]
            skill = np.unique(S[i])
            transi = np.random.binomial(n=1, p=self.transition[skill])
            L[i+1, skill] = np.maximum(transi, L[i, skill])
        # sample C matrix
        rel_L = np.take_along_axis(L, S, axis=1) # get relevant columns from L
        rel_L = np.min(rel_L, axis=1) # is any skill unlearned
        slip_mask = np.repeat(self.slip, int(n_exercises/self.n_blocks))
        guess_mask = np.repeat(self.guess, int(n_exercises/self.n_blocks))
        probs = rel_L*(1-slip_mask) + (1-rel_L)*guess_mask
        C = np.random.binomial(n=1, p=probs)
        return self._create_data_frame(L, S, C, student_id, max_skills)
        
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
    
    def _create_data_frame(self, L, S, C, student_id, max_skills):
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
        columns = ['Sk' + str(i) for i in range(self.n_skills)] + ["skill_name" + str(i) for i in range(max_skills)] + ['correct']
        data_frame = pd.DataFrame(data_frame, columns=columns).astype('int')
        data_frame['user_id'] = student_id
        return data_frame
    
    def _random_single(self, n_exercises, _):
        """Randomly samples the exercised skills for a single student's sequence.
        
        Arguments
        ---------
        n_exercises : int
            The number of exercises the sampled student completes.
        _ : arbitrary
            For consistency across sample functions.

        Returns
        -------
        (n_exercises, 1) ndarray
            The (single!) sampled skill for each exercise.
        """
        return np.random.randint(self.n_skills, size=(n_exercises, 1))
    
    def _blockwise(self, n_exercises, max_skills):
        """Samples the exercised skills for a single student's sequence from a block structure.

        Arguments
        ---------
        n_exercises : int
            The number of exercises the sampled student completes.
        n_blocks : int
            The number of blocks of exercises to be considered.
        max_skills : int
            The maximum number of skills that can appear in a single exercise.

        Returns
        -------
        (n_exercises, max_skills) ndarray
            Contains the skills concerned by each exercise for the student. One skill can occur multiple times in an exercise
        """
        sk_per_block = int(self.n_skills/self.n_blocks)
        ex_per_block = int(n_exercises/self.n_blocks)
        all_ex = np.empty((n_exercises, max_skills), dtype=int)
        for i in range(self.n_blocks):
            # sample skills
            first_block_skill = i * sk_per_block
            last_block_skill = (i+1) * sk_per_block
            sampled_skills = np.random.randint(first_block_skill, last_block_skill, (ex_per_block, max_skills))
            # overwrite output array
            block_start = i * ex_per_block
            block_end = (i+1) * ex_per_block
            all_ex[block_start:block_end] = sampled_skills 
        return all_ex