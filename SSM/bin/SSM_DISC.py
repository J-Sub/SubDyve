from scipy import stats
from typing import Iterable
from tqdm import tqdm
from rdkit.Chem import rdDepictor
import time
import pandas as pd
from collections import defaultdict
from math import log
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from mychem import *
from utils import *

import sys
import numpy as np
import rdkit
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
rdDepictor.SetPreferCoordGen(True)


class DILInew:
    def __init__(self, chemistry='graph', n_rw=10, n_alpha=0.5, iteration=10, pruning=False, n_walker=100, rw_mode="argmax"):
        self.chemistry, self.n_rw, self.n_alpha = chemistry, n_rw, n_alpha
        self.n_iteration, self.pruning, self.n_walkers = iteration, pruning, n_walker
        self.n_train, self.n_valid = 0, 0
        self.rw_mode = rw_mode
        #
        self.molinfo_df = pd.DataFrame()
        self.train_molinfo_df = pd.DataFrame()
        #
        # self.lfraglist, self.ledgelist = [], []
        # number of visited edges by RW for each graph; key: iteration, subkey: ltkbid, value: {edge_1:n_1, edge_2:n_2, ...}
        self.dEdgeUsedCount = defaultdict(dict)
        # key: iteration, subkey: ltkbid, value: {key: node, values: defaultdict(key 1: 'node_list', value 1: nodes (a list of lists) / key2: 'edge_list', value 2: edges (a list of lists))}
        self.dEdgelistUsage = defaultdict(dict)
        # fragment dictionary for each molecule; key: iteration, subkey: ltkbid, value: {frag_1:n_1, frag_2:n_2, ...}
        self.dNodeFragCount = defaultdict(dict)
        # key: iteration, subkey: ltkbid, value: {key: node (int), value: a list of fragments (list)}
        self.dNodeFragSmiles = defaultdict(dict)
        #
        # summary for class; Doublet preference; used only for training; key: iteration, subkey: fragment, value: dict(key: class, value: count (sum of the values in dEdgeUsedCount of the edges in the fragment))
        self.dEdgeClassDict = {}
        #
        # key: iteration, value: {ltkbid_0:T_0, ltkbid_1:T_1, ..., ltkbid_N:T_N} # ltkbid: 'ID', T: transition matrix of graph
        self.dMolTransDict = defaultdict(dict)
        # key: iteration, value: {ltkbid_0:F_0, ltkbid_1:F_1, ..., ltkbid_N:F_N}
        self.dMolPreferDict = defaultdict(dict)
        # self.dPreferDict = {} # key: iteration, value: Preference_iter
        #
        # key: iteration, value: a list of fragments that exist in only one class; used only for training
        self.lexclusivefrags = defaultdict(list)
        # key: iteration, value: a list of fragments that exist in both class; used only for training
        self.lunionfrags = defaultdict(list)
        # self.dFragSearch     = {}
        # self.prunehistory    = defaultdict(dict)

    def DoRandomWalk(self, n_iter, ltkbid, T):  # run_rw starts here
        # index: walker (int), column: atomID (int), value: visited nodes (list)
        dpaths = mychem.cal_path_df(
            self.molinfo_df["molgraph"][ltkbid], T, walkLength=self.n_rw, n_walker=self.n_walkers, mode=self.rw_mode)
        # dpaths: walklist(= list of nodes that a walker has gone through)
        #         {node1: visited_nodes, node2: visited_nodes}
        self.dEdgeUsedCount[n_iter][ltkbid], self.dEdgelistUsage[n_iter][ltkbid] = mychem.rwr_summary(
            self.molinfo_df["molgraph"][ltkbid], dpaths, n_walker=self.n_walkers)
        # dEdgeUsedCount: # of times edges used: {edge_1: 3, edge_2: 2, edge_3: 1, ...}
        # dEdgelistUsage: node: {node_list: [node_id, ..], edge_list: [edge_id, ..] }
        #                {0: {'node_list':[[0,1,2,,..], ..., []], 'edge_list': [['0_1', ...], ..., []]}
        self.dNodeFragCount[n_iter][ltkbid], self.dNodeFragSmiles[n_iter][ltkbid] = mychem.rw_getSmilesPathDict(
            mychem, self.molinfo_df["molobj"][ltkbid], self.dEdgelistUsage[n_iter][ltkbid])
        # dNodeFragCount: {frag: 3, frag:1, ...}, dNodeFragSmiles = {node:frag, ... }
    # END of DoRandomWalk

    def cal_preference(self, n_it):
        for ltkbid in self.dEdgeUsedCount[n_it]:
            nClass = self.molinfo_df["class"][ltkbid]

            for edge in self.dEdgeUsedCount[n_it][ltkbid]:
                a, b = list(map(int, edge.split('_')))
                frag = Chem.MolFragmentToSmiles(
                    self.molinfo_df["molobj"][ltkbid], atomsToUse=[a, b])

                try:
                    self.dEdgeClassDict[n_it][frag][nClass] += self.dEdgeUsedCount[n_it][ltkbid][edge]
                except:
                    try:
                        self.dEdgeClassDict[n_it][frag][nClass] = self.dEdgeUsedCount[n_it][ltkbid][edge]
                    except:
                        self.dEdgeClassDict[n_it][frag] = {}
                        self.dEdgeClassDict[n_it][frag][nClass] = self.dEdgeUsedCount[n_it][ltkbid][edge]
    # END of cal_preference

    def get_individual_F(self, n_iter, n_iter_pref_df, ltkbid, mode=False):
        def get_likelihood(mySeries):
            val = (mySeries[1] + 1e-8) / (mySeries[0] + 1e-8)
            return val

        F = np.zeros(self.dMolTransDict[n_iter-1][ltkbid].shape)

        for edge in self.molinfo_df["molgraph"][ltkbid].edges():
            n1, n2 = edge

            if self.molinfo_df["molgraph"][ltkbid][n1][n2]['order'] != 0:
                bond = self.molinfo_df["molobj"][ltkbid].GetBondBetweenAtoms(
                    n1, n2).GetIdx()
                frag_smi = Chem.MolFragmentToSmiles(
                    self.molinfo_df["molobj"][ltkbid], atomsToUse=[n1, n2], bondsToUse=[bond])

                try:
                    probSeries = n_iter_pref_df[frag_smi] / \
                        n_iter_pref_df.sum(axis=1)
                    F[n1, n2] = get_likelihood(probSeries)
                    F[n2, n1] = F[n1, n2]
                except:
                    F[n1, n2] = 0

        # normalize F
        for idx, row in enumerate(F):
            if row.sum() > 0:
                F[idx] = np.array([x/row.sum() for x in row])  # proportion
            else:
                continue

        F = F.transpose()

        return F
    # END of get_individual_F

    def rw_update_transitions(self, _T, _F, update_alpha):
        _T = _T * (1-update_alpha) + update_alpha * _F

        for idx, row in enumerate(_T):
            if row.sum() > 0:
                _T[idx] = np.array([x/row.sum() for x in row])  # proportion

        _T = _T.transpose()

        return _T
    # END of rw_update_transitions

    # START of get_fraglist
    def get_fraglist(self, n_iter):  # Get list of exclusive fragments
        #self.train_act_frag_df = pd.DataFrame(self.dNodeFragCount[n_iter], columns= self.dNodeFragCount[n_iter].keys())
        tempdf = pd.DataFrame(self.dNodeFragCount[n_iter], columns=self.dNodeFragCount[n_iter].keys(
        )).fillna(0).T  # row: ltkbid, column: fragment, value: count
        dfTempCnt = tempdf.merge(
            self.molinfo_df['class'], how='outer', left_index=True, right_on=self.molinfo_df.index)
        dfTempCnt.set_index(keys='key_0', inplace=True)
        nClassSpecificity = dfTempCnt.groupby(
            'class').any().astype(int).sum(axis=0) == 1
        # Fragments that exist in only one class
        lExList = nClassSpecificity[nClassSpecificity].index.to_list()
        nClassSpecificity = dfTempCnt.groupby(
            'class').any().astype(int).sum(axis=0) > 0
        # Fragments that exist in both classes
        lUnionList = nClassSpecificity[nClassSpecificity].index.to_list()
        return lExList, lUnionList
    # END of get_fraglist

    def search_fragments(self, sFrag):
        pdseries_search = pd.Series(
            1e-10, index=self.train_molinfo_df['class'].unique(), name=sFrag)

        for ltkbid in self.train_molinfo_df["ID"]:
            if self.train_molinfo_df["molobj"][ltkbid].HasSubstructMatch(Chem.MolFromSmarts(sFrag), useChirality=True):
                pdseries_search[self.train_molinfo_df["class"][ltkbid]] += 1

        pdseries_search = pdseries_search.divide(
            self.train_molinfo_df.groupby("class")["ID"].count() + 1e-10)

        return pdseries_search
    # END of search_fragments

    # START of DoPruning
    # MAIN HERE - argument: train_data
    def train(self, train_data):
        self.molinfo_df = train_data
        self.train_molinfo_df = train_data
        self.n_train = train_data.shape[0]
        print(f'Training...\nThe number of allowed walks: {self.n_rw}')

        for nI in range(self.n_iteration):  # iterate random walk process
            start = time.time()
            print(f'Start iteration {nI + 1} ----- ', end="")

            for ltkbid in self.molinfo_df["ID"]:  # iterate over molecules
                smiles = self.molinfo_df["SMILES"][ltkbid]

                # cal_T(molobj, molgraph, smiles, chemistry='graph') # first iteration
                if nI == 0:
                    T = mychem.cal_T(mychem, self.molinfo_df["molobj"][ltkbid], self.molinfo_df["molgraph"]
                                     [ltkbid], smiles, chemistry=self.chemistry)  # transition matrix of graph
                else:
                    # row: class, column: fragment, value: count
                    pd_pref = pd.DataFrame(
                        self.dEdgeClassDict[nI-1], columns=self.dEdgeClassDict[nI-1].keys()).fillna(0)
                    self.dMolPreferDict[nI - 1][ltkbid] = self.get_individual_F(nI, pd_pref, ltkbid)
                    T = self.rw_update_transitions(
                        self.dMolTransDict[nI-1][ltkbid], self.dMolPreferDict[nI-1][ltkbid], self.n_alpha)  # T * (1-alpha) + F * alpha

                self.dMolTransDict[nI][ltkbid] = T
                self.DoRandomWalk(nI, ltkbid, T)  # each molecule

            self.dEdgeClassDict[nI] = {}
            self.cal_preference(nI)  # Save Preference each iteration
            self.lexclusivefrags[nI], self.lunionfrags[nI] = self.get_fraglist(
                nI)
            fin = round((time.time() - start) / 60, 3)
            print(f'{self.n_rw} Random Walks completed in {fin} mins.')

        return self.dEdgeClassDict
    # END of train

    # MAIN HERE - argument: valid_data
    # , train_dFragSearch):
    def valid(self, valid_data, train_df, train_edgeclassdict):
        self.molinfo_df = valid_data
        self.train_molinfo_df = train_df
        self.n_valid = valid_data.shape[0]
        print(f'Test...\nThe number of allowed walks: {self.n_rw}')

        for nI in range(self.n_iteration):  # iterate random walk process
            start = time.time()
            print(f'Start iteration {nI + 1} ----- ', end="")

            for ltkbid in self.molinfo_df.index:  # iterate over molecules
                smiles = self.molinfo_df["SMILES"][ltkbid]

                if nI == 0:  # cal_T(molobj, molgraph, smiles, chemistry='graph')
                    T = mychem.cal_T(mychem, self.molinfo_df["molobj"][ltkbid], self.molinfo_df["molgraph"]
                                     [ltkbid], smiles, chemistry=self.chemistry)  # transition matrix
                else:
                    pd_pref = pd.DataFrame(
                        train_edgeclassdict[nI-1], columns=train_edgeclassdict[nI-1].keys()).fillna(0)
                    self.dMolPreferDict[nI-1][ltkbid] = self.get_individual_F(
                        nI, pd_pref, ltkbid, mode='test')
                    T = self.rw_update_transitions(
                        self.dMolTransDict[nI-1][ltkbid], self.dMolPreferDict[nI-1][ltkbid], self.n_alpha)  # T * (1-alpha) + F * alpha

                self.dMolTransDict[nI][ltkbid] = T
                self.DoRandomWalk(nI, ltkbid, T)  # each molecule

            fin = round((time.time() - start) / 60, 3)
            print(f'{self.n_rw} Random Walks completed in {fin} mins.')


class analyze_individual():
    def __init__(self):
        # key: iteration, value: pd.DataFrame (row: ltkbid, column: fragment, value: count)
        self.frag_df = defaultdict(pd.DataFrame)
        # key: iteration, value: a list of fragments
        self.lfraglist = defaultdict(list)
        self.ledgelist = defaultdict(list)

    def get_frag_df(self, srw, iteration, set_type):
        self.frag_df[iteration] = pd.DataFrame(
            srw.dNodeFragCount[iteration], columns=srw.dNodeFragCount[iteration].keys())
        self.frag_df[iteration] = self.frag_df[iteration].fillna(0).T
        self.lfraglist[iteration] = list(set(self.frag_df[iteration].columns))
        print(
            f'The number of fragments in the {set_type} data during the iteration {iteration + 1}: {len(self.lfraglist[iteration])}')

    def get_edge_df(self, iteration=0):
        self.ledgelist = list(set(self.dEdgeClassDict[iteration].keys()))
        print(
            f'The number of edges for iteration {iteration}: {len(self.ledgelist[iteration])}')


def prepare_classification(df, molinfo):
    #df_binary = df > 0
    df_binary = df
    df_binary = df_binary.astype(int)
    df_new = df.merge(molinfo['class'], how='outer',
                      left_index=True, right_on=molinfo["ID"])
    df_new = df_new.set_index("key_0", drop=True)
    df_new.index.name = None
    df_new = df_new.fillna(0)

    df_entropy = cal_entropy_subgraph(df_new, 'Train')

    X = df_new.drop('class', axis=1)
    y = df_new['class']
    return X, y, df_entropy


def prediction(train_obj, valid_obj, nIter, output_dir, train_molinfo_df, valid_molinfo_df, n_seed=0, DiSC=0, best_Iter=16):  # train.pickle, test.pickle
    print(f"\nTotal Iteration: {nIter}")
    print(f'==The best iteration for prediction: {best_Iter}==')
    pd_result = pd.DataFrame(0, index=range(nIter),  columns=[
                             'n_union_subgraphs', 'n_train_subgraphs', 'n_valid_subgraphs', 'Accuracy', 'BAcc', 'Precision', 'Recall', 'F1_score', 'AUC', 'MCC'], dtype=np.float64)
    pd_confusion = pd.DataFrame(0, index=range(
        nIter), columns=["tn", "fp", "fn", "tp"])
    analyze_train = analyze_individual()
    analyze_valid = analyze_individual()

    for nI in range(nIter):
        print(f'\nIteration {nI + 1}')
        analyze_train.get_frag_df(train_obj, iteration=nI, set_type='training')
        analyze_valid.get_frag_df(valid_obj, iteration=nI, set_type='test')
        train_mat = analyze_train.frag_df[nI]
        valid_mat = analyze_valid.frag_df[nI]
        # features
        n_train = train_mat.shape[1]
        n_valid = valid_mat.shape[1]
        # merged_mat = train_mat.append(valid_mat, sort=False).fillna(0)
        merged_mat = pd.concat([train_mat, valid_mat], sort=False).fillna(0)
        n_union = merged_mat.shape[1]
        train_mat = merged_mat.iloc[:train_mat.shape[0], :n_train]
        valid_mat = merged_mat.iloc[train_mat.shape[0]:, :n_train]
        train_X, train_y, df_entropy_train = prepare_classification(
            train_mat, train_molinfo_df)
        valid_X = valid_mat
        if 'class' in valid_molinfo_df.columns:
            df_entropy_valid = cal_entropy_subgraph(
                pd.concat([valid_mat, valid_molinfo_df['class']], axis=1), 'Valid')
            df_entropy = pd.merge(df_entropy_train, df_entropy_valid,
                                  how='outer', left_index=True, right_index=True)
        else:
            df_entropy = df_entropy_train

        print("Subgraph matrix generation is finished. Start making predictios.")
        print(f'Training/Test data shape: {train_X.shape} / {valid_X.shape}')
        # performance
        smi_rf = RFC(random_state=n_seed)  # seed number here
        smi_rf.fit(train_X, train_y)
        df_entropy['Importance'] = smi_rf.feature_importances_
        df_entropy = df_entropy.sort_values('Importance', ascending=False)
        rf_preds = smi_rf.predict(valid_X)
        rf_probs = smi_rf.predict_proba(valid_X)[:, 1]

        if 'class' in valid_molinfo_df.columns:
            rf_true = valid_molinfo_df['class'].to_list()

            accuracy = accuracy_score(rf_true, rf_preds)
            balanced_accuracy = balanced_accuracy_score(rf_true, rf_preds)
            precision = precision_score(rf_true, rf_preds)
            recall = recall_score(rf_true, rf_preds)
            f1 = f1_score(rf_true, rf_preds)
            try:
                roc_auc = roc_auc_score(rf_true, rf_probs)
            except ValueError:
                roc_auc = np.nan
            mcc = matthews_corrcoef(rf_true, rf_preds)
            confusion_mat = confusion_matrix(rf_true, rf_preds).ravel()

            result_list = [n_union, n_train, n_valid, accuracy,
                           balanced_accuracy, precision, recall, f1, roc_auc, mcc]
            pd_result.iloc[nI] = result_list
            pd_confusion.iloc[nI] = confusion_mat

        pd_output = valid_obj.molinfo_df.loc[:, ['SMILES']]
        pd_output['prediction'] = rf_preds.tolist()
        pd_output['probability'] = rf_probs.tolist()
        os.makedirs(f'{output_dir}/iteration_{nI+1}', exist_ok=True)
        fname = f'{output_dir}/iteration_{nI+1}/predictions.csv'
        pd_output.to_csv(fname)
        df_entropy.to_csv(
            f'{output_dir}/iteration_{nI+1}/subgraph.csv')
        df_entropy_important = df_entropy[(df_entropy['Entropy (Train)'] < 0.5) & (
            df_entropy['Importance'] > 0.0001)]
        df_entropy_important.to_csv(
            f'{output_dir}/iteration_{nI+1}/subgraph_important.csv', float_format='%.4f')
        df_SA = df_entropy_important[df_entropy_important['Support (Train); F_1'] > (
            df_entropy_important['Support (Train); F_0'] + 0.01)]
        df_SA.to_csv(f'{output_dir}/iteration_{nI+1}/subgraph_SA.csv', float_format='%.4f')

        if DiSC not in [0, 1] and (nI+1) == best_Iter: 
            for k in range(2, DiSC+1):
                if os.path.exists(f'{output_dir}/iteration_{nI+1}/DiSC_{k}.csv'):
                    continue
                result_df = SMARTS_pattern_mining(valid_obj, train_X, k=k)
                if result_df is not None:
                    result_df.index.name = 'DiSC'
                    result_df.to_csv(
                        f'{output_dir}/iteration_{nI+1}/DiSC_{k}.csv', float_format='%.4f')

    if 'class' in valid_molinfo_df.columns:
        pd_result.index.name = 'Iteration'
        pd_result.columns.name = 'Iteration'
        pd_result.index = range(1, nIter+1)
        pd_confusion.index.name = 'Iteration'
        pd_confusion.columns.name = 'Iteration'
        pd_confusion.index = range(1, nIter+1)
        int_col = {'n_union_subgraphs': int,
                   'n_train_subgraphs': int, 'n_valid_subgraphs': int}
        pd_result = pd_result.astype(int_col)
        pd_result.index.name = 'iteration'
        pd_result.to_csv(f'{output_dir}/performance.csv', float_format='%.4f')
        pd_confusion.to_csv(f'{output_dir}/confusion_matrix.csv')

    print(f'\nResult files are saved in {output_dir}.')
    # END of classification


def SMARTS_pattern_mining(valid_obj, train_X, k=2, epsilon=0.01, minimum_support_percent=2, minimum_entropy_cutoff=0.1):
    minimum_support = train_X.shape[1] * minimum_support_percent / 100

    train_labels = valid_obj.train_molinfo_df['class']
    train_mols = valid_obj.train_molinfo_df['molobj']

    def match(train_mols: Iterable[rdkit.Chem.rdchem.Mol], fragment: rdkit.Chem.rdchem.Mol):
        return np.array([m.HasSubstructMatch(fragment) for m in train_mols])

    fragment_set = np.array(train_X.columns)

    fragment_set = list(map(lambda x: Chem.MolFromSmarts(x), fragment_set))
    support_unique = dict()
    for i, fragment in enumerate(tqdm(fragment_set)):
        if np.count_nonzero(train_X.values[:, i]) > minimum_support:
            to_tuple = tuple(np.bool_(train_X.values[:, i]))
            if to_tuple not in support_unique:
                support_unique[to_tuple] = fragment
            else:
                if support_unique[to_tuple].GetNumAtoms() > fragment.GetNumAtoms():
                    support_unique[to_tuple] = fragment

    fragment_set = list(support_unique.values())
    print(f"\n{len(fragment_set)} fragments remained after support-drop")

    num_pos, num_neg = (train_labels == 1).sum(), (train_labels == 0).sum()

    def support(match_result):
        pos_support = (((train_labels == 1)*(match_result == True))).sum()
        neg_support = (((train_labels == 0)*(match_result == True))).sum()
        return pos_support, neg_support

    def entropy(match_result, return_support=False):  # significance
        pos_support, neg_support = support(match_result)
        if pos_support + neg_support < minimum_support:
            if return_support:
                return 0, (pos_support, neg_support)
            else:
                return 0
        pos_ratio = (pos_support / num_pos) + epsilon
        neg_ratio = (neg_support / num_neg) + epsilon
        tot = pos_ratio + neg_ratio
        pos_ratio /= tot
        neg_ratio /= tot
        if return_support:
            return (1 + pos_ratio * np.log2(pos_ratio) + neg_ratio * np.log2(neg_ratio)), (pos_support, neg_support)
        else:
            return (1 + pos_ratio * np.log2(pos_ratio) + neg_ratio * np.log2(neg_ratio))

    def entropy_evaluate(smarts, return_support=False):
        f = Chem.MolFromSmarts(smarts)
        return entropy(match(train_mols, f), return_support)

    fragment_smarts_set = list(
        map(lambda x: Chem.MolToSmarts(x), fragment_set))
    fragment_with_smarts = [(a, b)
                            for (a, b) in zip(fragment_set, fragment_smarts_set)]
    maximum_num_atoms = max(list(map(lambda x: x.GetNumAtoms(), train_mols)))

    entropy_scores = np.array([entropy_evaluate(b)
                              for (a, b) in fragment_with_smarts])
    iterate_count = 0
    result_dict = {}
    tqdm_progressbar = tqdm(total=len(fragment_with_smarts))
    required_subset_len = k

    def backtrack(A, subset, subset_index, index, current_ent):
        nonlocal result_dict, iterate_count, required_subset_len
        if len(subset) > required_subset_len:
            return
        if len(subset) > 0:
            current_smarts = ".".join([b for (a, b) in subset])
            current_fragment = Chem.MolFromSmarts(current_smarts)
            atom_cnt = current_fragment.GetNumAtoms()
            if atom_cnt > maximum_num_atoms:
                return
            ent = entropy_evaluate(current_smarts, return_support=True)
            iterate_count += 1
            tqdm_progressbar.set_postfix_str(
                f"Seen iteration : {iterate_count}")
            if sum(ent[1]) < minimum_support:
                return
            if len(subset) > 1 and ent[0] <= max(minimum_entropy_cutoff, current_ent, np.max(entropy_scores[subset_index])):
                return
            current_ent = ent[0]
            if len(subset) == required_subset_len:
                result_dict[current_smarts] = ent
                print(
                    f"{iterate_count} {subset_index} : {current_smarts} Score = {ent}", flush=True)
                if (iterate_count % 5000) == 0:
                    print(
                        f"{iterate_count} {subset_index} : {current_smarts} Score = {ent}", file=sys.stderr)
        if len(subset) == 1:
            tqdm_progressbar.update(1)
            tqdm_progressbar.set_postfix_str(
                f"Seen iteration : {iterate_count}")
        for i in range(index, len(A)):
            subset.append(A[i])
            subset_index.append(i)
            backtrack(A, subset, subset_index, i + 1, current_ent)
            subset.pop(-1)
            subset_index.pop(-1)
        return

    subset, subset_index = [], []
    index = 0
    backtrack(fragment_with_smarts, subset, subset_index, index, 0.0)

    if len(result_dict) == 0:
        return None

    result_df = pd.DataFrame(result_dict).T
    result_df.columns = ['significance', 'support']
    result_df[['support F_1', 'support F_0']] = pd.DataFrame(
        result_df['support'].to_list(), index=result_df.index) / (num_pos, num_neg)
    result_df.drop(columns='support', inplace=True)
    result_df.sort_values('significance', ascending=False, inplace=True)

    return result_df
