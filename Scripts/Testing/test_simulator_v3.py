"""
Tests whether the simulator_v3 works as expected by computing the empirical transition and prior probabilites from simulated data
"""
import numpy as np
from simulator import Simulator_v3


simulator_params = {
    'blocks': [5],
    "n_skills": 5 # sum('blocks')
}

bkt_params = {
    'l0': (0, 0, 0, 0, 0),
    'transition': (0.1, 0.2, 0.3, 0.4, 0.5),
    'slip': (0.2,),
    'guess': (0.2,),
    'max_skills': 3
}

sampling_params = {
    'n_students': 10000,
    'n_exercises': 60 
}


# reproducibility seed
np.random.seed(2021)


if __name__ == '__main__':
    # simulate data
    student_simulator = Simulator_v3(**simulator_params, **bkt_params)
    data = student_simulator.sample_students(**sampling_params)
    data['next_user_id'] = data['user_id'].shift(-1)

    # transition check
    for i in range(len(bkt_params['transition'])):
        interesting_rows = (data['skill_name' + str(i)] == 1) & \
                           (data['Sk' + str(i)] == 0) & \
                           (data['user_id'] == data['next_user_id'])
        interesting_rows = interesting_rows.shift(1).fillna(False)
        empirical_transition = data.loc[interesting_rows]['Sk' + str(i)].mean()
        true_transition = bkt_params['transition'][i]
        print(f'Skill {i} ... {true_transition} ... {empirical_transition}')

    # prior check
    first_row_per_student = data.groupby('user_id').first()
    state_columns = [c for c in first_row_per_student.columns if c.startswith('Sk')]
    print(np.all(first_row_per_student[state_columns] == 0))