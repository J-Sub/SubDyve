import argparse
import os
import random
import sklearn
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings(action='ignore')

# set up seed value
def seed_everything(seed=0):
	random.seed(seed)
	np.random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	sklearn.utils.check_random_state(seed)

def float_range(arg):
    try:
        value = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{arg} is not a valid float value")
    
    if value < 0.0 or value > 1.0:
        raise argparse.ArgumentTypeError(f"{arg} is not in the range [0.0, 1.0]")
    
    return value

def print_changed_args(args_trained, args_input):
    if args_trained.n_rw != args_input.rw:
        print(f'The value of argument "rw changes from "{args_input.rw}" to "{args_trained.n_rw}" according to the trained model.')
    if args_trained.n_alpha != args_input.alpha:
        print(f'The value of argument "rw changes from "{args_input.alpha}" to "{args_trained.n_alpha}" according to the trained model.')
    if args_trained.n_iteration != args_input.iterations:
        print(f'The value of argument "rw changes from "{args_input.iterations}" to "{args_trained.n_iteration}" according to the trained model.')
    if args_trained.n_walkers != args_input.nWalker:
        print(f'The value of argument "rw changes from "{args_input.n_walkers}" to "{args_trained.nWalker}" according to the trained model.')

def cal_entropy_subgraph(df, split_type):
    df_0 = df[df['class'] == 0].copy()
    df_1 = df[df['class'] == 1].copy()
    df_0.drop(columns='class', inplace=True)
    df_1.drop(columns='class', inplace=True)

    support_0 = df_0.sum()/ df_0.shape[0]
    support_1 = df_1.sum()/ df_1.shape[0]

    entropy_subgraph = pd.Series(stats.entropy([support_0, support_1], base=2), index=support_0.index)
    df_entropy = pd.concat([support_0, support_1, entropy_subgraph], axis=1)
    df_entropy.index.name = 'subgraph'
    df_entropy.columns = [f'Support ({split_type}); F_0', f'Support ({split_type}); F_1', f'Entropy ({split_type})']

    return df_entropy

