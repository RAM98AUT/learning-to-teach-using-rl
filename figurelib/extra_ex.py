import numpy as np


class ExtraEx:
    """Implementation for the experiment of BKT simulated students receiving a fixed amount of exercises
    when looking at two different learning groups of same size

    Parameters
    ----------
    learning_array_slow : ndarray (slow students,)
        provides when each simulated students of the slow group learned a skill
    learning_array_fast : ndarray (fast students)
        provides when each simulated students of the fast group learned a skill

    """
    
    def __init__(self,learning_array_slow, learning_array_fast):
        """
        Inits a class with 2 learning arrays and size of both student groups
        """
        self.learning_array_slow = learning_array_slow
        self.learning_array_fast = learning_array_fast
        self.slow_stud = self.learning_array_slow.shape[0]
        self.fast_stud = self.learning_array_fast.shape[0]
        
    def graph_data(self,min_num_ex = 1,max_num_ex = 100):
        """
        Calculates the average number of extra opportunities and the learning gap 
        between slow/fast group for the range between min and max num of exercises
        
        Parameters
        ----------
        min_num_ex : int, optional
            Minimum number of exercise students receive, the default is 1.
        max_num_ex : int, optional
            Maximum number of exercise students receive, must be larger than min_num_ex
            the default is 100. 

        Returns
        -------
        av_ex_op : list (max_num_ex-min_num_ex+1,)
            average number of extra opportunities after skill is already learned
        gap : list (max_num_ex-min_num_ex+1,)
            learning gap between slow and fast learners after a fixed number of exercises

        """
        av_ex_op =[]
        gap = []
        for ex in range(min_num_ex,max_num_ex+1):
            slow_learned = self.learning_array_slow[np.where(self.learning_array_slow <= ex)] 
            fast_learned = self.learning_array_fast[np.where(self.learning_array_fast <= ex)] 
            gap.append(((1 - slow_learned.shape[0]/self.slow_stud) - (1- fast_learned.shape[0]/self.fast_stud))*100)
            learned = np.concatenate((slow_learned,fast_learned))
            av_ex_op.append(0 if learned.shape[0]==0 else sum(ex - learned)/learned.shape[0])  
        return av_ex_op, gap