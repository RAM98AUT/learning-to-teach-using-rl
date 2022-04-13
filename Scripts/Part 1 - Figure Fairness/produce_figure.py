"""
This script runs the simiulator for two separate groups 
(slow/fast learners) and produces the graph that we should replicate
Programmed for 200 students and 1 skill
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from simulator import Simulator
from figurelib import BKTExecutor, ExtraEx


np.random.seed(11234)
save_data = False

# common arguments
n_skills = 1
n_exercises = 140

# number of students
n_slow = 100
n_fast = 100

# performance parameters
slow_params = {
    'l0': 0,
    'transition': 0.1,
    'slip': 0.1,
    'guess': 0.15
}

fast_params = {
    'l0': 0,
    'transition': 0.3,
    'slip': 0.2,
    'guess': 0.2
}

# bkt mixed
bkt_params = {
    'prob_l0': 0.0357,
    'prob_trans': 0.0878,
    'prob_guess': 0.1528,
    'prob_slip': 0.2580
}


if __name__=='__main__':
    # create simulator instances for slow and fast learners
    slow_student_simulator = Simulator(n_skills, **slow_params)
    fast_student_simulator = Simulator(n_skills, **fast_params, min_id=n_slow)

    # sample and concatenate data
    data_slow, l_arr_slow = slow_student_simulator.sample_students(n_students=n_slow, n_exercises=n_exercises)
    data_fast, l_arr_fast = fast_student_simulator.sample_students(n_students=n_fast, n_exercises=n_exercises)
    data_complete = pd.concat([data_slow, data_fast])

    # save data in data folder if required
    if save_data:
        data_slow.to_csv("data/data_slow.csv", index=False)
        data_fast.to_csv("data/data_fast.csv", index=False)
        data_complete.to_csv("data/data_complete.csv", index=False)

    # Rebuild graph from fairness paper
    # Extra Exercises
    fixed_op = ExtraEx(l_arr_slow, l_arr_fast)
    ex_op, gap = fixed_op.graph_data()

    # Apply fitted bkt model
    run_BKT = BKTExecutor(**bkt_params)
    gap_bkt, ex_op_bkt = run_BKT.run_for_graph([data_slow, data_fast], [[0, 100], [100, 200]], [np.ravel(l_arr_slow), np.ravel(l_arr_fast)]) 

    # Plot graph
    plt.plot(ex_op,gap, "bo", label="Fixed number of practise opportunities")
    plt.plot(ex_op_bkt, gap_bkt, c="orange", marker="o", label="BKT-mixed")
    plt.plot([ex_op_bkt,100], [gap_bkt,gap_bkt], "k", linewidth=0.8)
    plt.xlabel("Average number of extra opportunities")
    plt.ylabel("Equity gap")
    plt.legend(loc="upper right")
    plt.savefig("output/Figure that we should reproduce (see M2).png")  