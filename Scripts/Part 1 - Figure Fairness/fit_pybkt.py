"""
This script uses the model pyBKT in order to compare results with our own
"""
import pandas as pd
from pyBKT.models import Model


# Data
data_complete = pd.read_csv('data/data_slow.csv', encoding='latin')

# Model initialisation
model_params = {
    'seed': 123,
    'num_fits': 1
}

fit_params = {
    'multigs': False,
    'multilearn': False,
    'forgets': False
}


if __name__ ==  '__main__':
    model = Model(**model_params)
    model.fit(data=data_complete, **fit_params)
    print(model.coef_)