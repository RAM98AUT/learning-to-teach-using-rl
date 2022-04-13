import numpy as np


class BKTExecutor:
    """Implementation for the experiment of BKT simulated students.

    For one skill two different datasets can be analysed

    Attributes
    ----------
    BKT parameters (L0, P(G), P(S), P(T))
    prob_l0 : float
        The probability that the skill is in the learned state at the beginning.
    prob_guess : float
        The probability of guessing the correct answer when the skill is unlearned.
    prob_slip : float
        The probability of giving an incorrect answer when the skill is learned.
    prob_trans : float
        The probability of a transition from unlearned to learned.
    mastery_threshold: float
        If P(L_n) > mastery_threshold, we assume the skill is learned.
    """
    
    def __init__(self, prob_l0, prob_guess, prob_slip, prob_trans, mastery_threshold=0.95):
        """Inits a probabilistic BKT executor with necessary BKT parameters and mastery_threshold."""
        self.prob_l0 = prob_l0
        self.prob_guess = prob_guess
        self.prob_slip = prob_slip
        self.prob_trans = prob_trans
        self.mastery_threshold = mastery_threshold
    
    def run_for_graph(self, data, num_stud_range, learning_arrays):
        """
  

        Parameters
        ----------
        data : Dataframe
            complete dataframe that is used for calculations
        num_stud_range : list(list(lower,upper),list(lower,upper))
            provides the range for both data sets concerning student id's
        learning_arrays : list [array1,array2]
            list that includes the learning array of both groups

        Returns
        -------
        gap : int
            difference between the percentage of students that learned the skill in both groups
        num_ex : int
            average number of average opportunities

        """
        perc_not_learned = list()
        extra_ex = list()
        for data, students, arr in zip(data, num_stud_range, learning_arrays):
           learned_when = list()
           for stud in range(students[0], students[1]):
               learned_when.append(self._execute_student(data, stud))
           perc_not_learned.append(1-np.sum(arr<=np.asarray(learned_when)) / (students[1]-students[0]))
           subt_list = learned_when - arr
           extra_ex += [x for x in subt_list if x>=0]
        gap = (perc_not_learned[0]-perc_not_learned[1]) * 100
        num_ex = sum(extra_ex) / len(extra_ex)
        return gap, num_ex
        
    def _execute_student(self, data, stud):
        """
        Calculates with given exercises and their correctness the probability 
        that a skill is learned till the mastery threshold is exceeded

        Parameters
        ----------
        data : Dataframe
            complete dataframe that is used for calculations
        stud : int
            id of the student

        Returns
        -------
        step : int
            returns after how many steps the algorithm assumes that the student has learned the skill

        """
        prob = self.prob_l0
        step = 0
        data_stud = data.loc[data['user_id'] == stud]
        while prob < self.mastery_threshold:
            evidence = data_stud.loc[data_stud["order_id"] == step, 'correct'].values[0]
            if evidence == 0:
                prob_under_ev = (self.prob_slip * prob) / (self.prob_slip * prob + (1-self.prob_guess) *(1-prob))
            else:
                prob_under_ev = ((1-self.prob_slip) * prob) / ((1-self.prob_slip) * prob + self.prob_guess *(1-prob))  
            prob = prob_under_ev + (1-prob_under_ev) * self.prob_trans
            step += 1
        return step