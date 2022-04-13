import numpy as np
import pandas as pd


class Simulator:
    """
    A class to simulate students with a common performance from a BKT model.
    Each exercises includes exactly one skill, each skill can have their own prior,
    all other parameters (slip,guess,transition) are shared

    Attributes
    ----------
    n_skills : int
        Number of different skills to be considered.
    l0 : float or (n_skills,) array-like
        The priors for the skills. Can be either float for a common prior.
        Or a np.array/tuple/list of length n_skills for skill-individual priors.
    transition : float
        The transition probability.
    slip : float
        The slip probability.
    guess : float
        The guess probability.
    min_id : int (default=0)
        The user_id of the first student to be sampled.
        The consecutive students have user_id min_id+1, min_id+2, ...
    """
    
    def __init__(self, n_skills, l0, transition, slip, guess, min_id=0):
        """Inits a simulator with args n_skills, l0, transition, slip, guess and min_id."""
        self.n_skills = n_skills
        self.l0 = l0
        self.transition = transition
        self.slip = slip
        self.guess = guess
        self.min_id = min_id
        
    def sample_students(self, n_students, n_exercises):
        """Simulates sequences for multiple students
        
        Arguments
        ---------
        n_students : int
            The number of students to simulate.
        n_exercises : int 
            The number of exercises to simulate for each student.

        Returns
        -------
        DataFrame
            DataFrame containing all sampled information with n_students*n_exercises rows.
            Basically the vertical concatenation of multiple outputs from the _sample_student method.
        (n_students*n_exercises,5) ndarray
        Learning array: ndarray(n_skills)
            Containing the time for each student skill combination when the skill was learned.
        """
        # init output objects
        students = []
        learning_time = [] # array for saving when skills were learned
        # sample students
        for student_id in range(self.min_id, self.min_id + n_students):
            df, learning = self._sample_student(n_exercises, student_id)
            students.append(df)
            learning_time.append(learning)
        # process and return sampling results
        learning_array = np.array(learning_time) 
        students = pd.concat(students).reset_index().rename(columns={'index': 'order_id'})
        return students, learning_array
        
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
        list
            A list of length n_skills with the time when each skill was learned.
            inf for unlearned skills.
        """
        # sample initial L and S matrix
        L = self._init_L(n_exercises)
        S = np.random.randint(self.n_skills, size=n_exercises)
        learning = [np.inf if x == 0 else 0 for x in L[0, :]]
        # update L matrix
        for i in range(n_exercises-1):
            skill = S[i]
            L[i+1, :] = L[i, :]
            if L[i+1, skill] == 0:
                transi = np.random.binomial(n=1, p=self.transition)
                L[i+1, skill] = transi
                if transi == 1: 
                    learning[skill] = i+1 
            else:
                L[i+1, skill] = 1  
        # sample C matrix
        probs = L*(1-self.slip) + (1-L)*self.guess
        probs = probs[range(n_exercises), S]
        C = np.random.binomial(n=1, p=probs, size=n_exercises)
        return self._create_data_frame(L, S, C, student_id), learning
        
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

        Returns
        -------
        DataFrame 
            Basically a horizontal concatenation of the arguments.
            Columns: [Sk0, Sk1, Sk2, ..., skill_name, correct, user_id]         
        """
        data_frame = np.c_[L, S, C]
        columns = ['Sk' + str(i) for i in range(self.n_skills)] + ['skill_name', 'correct']
        data_frame = pd.DataFrame(data_frame, columns=columns).astype('int')
        data_frame['user_id'] = student_id
        return data_frame