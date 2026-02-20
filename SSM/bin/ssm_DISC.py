# Read Python libraries
import os
import argparse
import pickle
import pandas as pd

# Read base class
from mychem import *
from mydata import PrepareData
from SSM_DISC import DILInew, prediction
from utils import *

class get_subgraph():
    def __init__(self, args, chemistry='atom', pruning='pure', rule='random'):
        self.trained = '' # model
        self.test = '' # model
        self.rw = args.rw
        self.chemistry = chemistry
        self.alpha = args.alpha
        self.iterations = args.iterations
        self.pruning = pruning
        self.nWalker = args.nWalker
        self.sRule = rule
        self.nSeed = args.seed
        #
        self.train_df         = pd.DataFrame()
        self.train_molinfo_df = pd.DataFrame()
        self.test_df          = pd.DataFrame()
        self.test_molinfo_df  = pd.DataFrame()

    def read_model(self, model_file):
        with open(model_file, 'rb') as trained_file:
            self.trained = pickle.load(trained_file)
        print("\nTrained model loaded\n")

    def read_data(self, train_data, test_data, train=True):
        my_data = PrepareData()
        my_data.read_data(train_fname=train_data, test_fname=test_data, train=train)
        self.train_df = my_data.train_df
        self.test_df = my_data.test_df
        if train == True:
            self.train_df, self.train_molinfo_df = my_data.prepare_rw_train(self.train_df)
        else:
            self.train_molinfo_df = self.trained.train_molinfo_df
        self.test_df, self.test_molinfo_df = my_data.prepare_rw(self.test_df)
        print("Data Preraparation Complete\n")

# Read data
def SSM_parser():
    parser = argparse.ArgumentParser(description = 'Supervised Subgraph Mining')

    parser.add_argument('--train_data', default=None, type=str)
    parser.add_argument('--test_data', required=True, type=str)
    parser.add_argument('--output_dir', default='./results', type=str)
    parser.add_argument('--trained_file', default=None, type=str)
    parser.add_argument('--rw', '-l', default=7, type=int)
    parser.add_argument('--alpha', '-a', default=0.1, type=float_range)
    parser.add_argument('--iterations', '-k', default=20, type=int)
    parser.add_argument('--nWalker',  default=5, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--DiSC', default=2, type=int)
    parser.add_argument('--bestIteration', default=16, type=int)

    args = parser.parse_args()
    return args

def main(args):
    print(f"\nCurrent working directory: {os.getcwd()}\n")

    # Load main SSM class
    ssm = get_subgraph(args)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load trained model
    if args.trained_file is not None:
        ssm.read_model(args.trained_file)
        ssm.read_data(args.train_data, args.test_data, train=False)

        print_changed_args(ssm.trained, ssm)
        ssm.chemistry = ssm.trained.chemistry
        ssm.rw = ssm.trained.n_rw
        ssm.alpha = ssm.trained.n_alpha
        ssm.iterations = ssm.trained.n_iteration
        ssm.pruning = ssm.trained.pruning
        ssm.nWalker = ssm.trained.n_walkers
        ssm.sRule = ssm.trained.rw_mode
    # Run Supervised Subgraph Mining for the training data
    else:
        ssm.read_data(args.train_data, args.test_data, train=True)
        ssm.train = DILInew(chemistry = ssm.chemistry, n_rw = ssm.rw, n_alpha = ssm.alpha, iteration = ssm.iterations, pruning = ssm.pruning, n_walker = ssm.nWalker , rw_mode = ssm.sRule)
        seed_everything(args.seed)
        ssm.train.train(ssm.train_molinfo_df)
        with open(f'{args.output_dir}/ssm_train.pickle', 'wb') as train_archive:
            pickle.dump(ssm.train, train_archive, pickle.HIGHEST_PROTOCOL)
        ssm.read_model(f'{args.output_dir}/ssm_train.pickle')

    # Run Supervised Subgraph Mining for the test data
    ssm.test = DILInew(chemistry = ssm.chemistry, n_rw = ssm.rw, n_alpha = ssm.alpha, iteration = ssm.iterations, pruning = ssm.pruning, n_walker = ssm.nWalker , rw_mode = ssm.sRule)
    seed_everything(args.seed)
    ssm.test.valid(ssm.test_molinfo_df, ssm.train_molinfo_df, ssm.trained.dEdgeClassDict)
    
    with open(f'{args.output_dir}/ssm_test.pickle', 'wb') as valid_archive:
        pickle.dump(ssm.test, valid_archive, pickle.HIGHEST_PROTOCOL)
    prediction(ssm.trained, ssm.test, ssm.iterations, args.output_dir, ssm.train_molinfo_df, ssm.test_molinfo_df, ssm.nSeed, args.DiSC, args.bestIteration)

if __name__ == '__main__':
    # Initialize and read trained model
    args = SSM_parser()

    main(args)
