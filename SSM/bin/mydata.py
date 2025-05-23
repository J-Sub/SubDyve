import os
from collections import defaultdict
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from pysmiles import read_smiles
import logging
logging.getLogger('pysmiles').setLevel(logging.CRITICAL)


class myobj():
    def __init__(self, sanitize=True):
        self.original_data  = pd.DataFrame() 
        self.train_df       = pd.DataFrame() 
        self.test_df        = pd.DataFrame()
        self.sanitize       = sanitize

    def sanitize_mols(self, row):
        remover = SaltRemover()
        mol = Chem.MolFromSmiles(row["SMILES"])
        ChemMol = remover.StripMol(mol)

        try:
            smiles = Chem.MolToSmiles(ChemMol, isomericSmiles=True, canonical=True, rootedAtAtom=-1)
        except:
            smiles = Chem.MolToSmiles(ChemMol, isomericSmiles=True, canonical=True)

        row['SMILES'] = smiles

        return row

    def read_data(self, train_fname, test_fname, train=True):
        if train_fname is None and train == True:
            print(f"Training Data: DILIst from (Chem Res Toxicol, 2021)")
            self.original_data = pd.read_csv("training_data/deepdili_dilist_train_all.csv", header="infer")
        elif train_fname is not None and train == True:
            print(f'Training Data: {train_fname}')
            self.original_data = pd.read_csv(train_fname)
            

        if train == True:
            self.original_data = self.original_data[~self.original_data["SMILES"].isna()]
            self.train_df  = self.original_data[["SMILES", "label"]]
            self.train_df.columns = ["SMILES", "class"] 

        
        print(f"Test Data: {test_fname}")
        self.test_df = pd.read_csv(test_fname, header="infer")

        def run_sanitize(df):
            if self.sanitize:
                df   = df.apply(self.sanitize_mols, axis=1)
            return df

        self.train_df = run_sanitize(self.train_df)
        self.test_df  = run_sanitize(self.test_df)
        self.train_df.reset_index(inplace=True, drop=True)
        self.test_df.reset_index(inplace=True, drop=True)

        return self.train_df, self.test_df

class PrepareData():
    def __init__(self, sanitize=True, valid_external=False, mode = "cv"):
        self.mydata          = myobj(sanitize=sanitize)
        self.mode            = mode
        self.train_df, self.test_df = pd.DataFrame(), pd.DataFrame()

    def read_data(self, train_fname, test_fname, train=True):
        self.train_df, self.test_df = self.mydata.read_data(train_fname=train_fname, test_fname = test_fname, train=train)
        if train == True:
            print(f'Training Data shape: {self.train_df.shape}')
        print(f'Test Data shape: {self.test_df.shape}\n')

    def prepare_rw(self, data): 
        removed    = []

        if 'label' not in data.columns:
            molinfo_list = []
            columns = ["Number", "SMILES", "molobj", "molgraph"]
        else:
            molinfo_list = []
            columns = ["Number", 'class', "SMILES", "molobj", "molgraph"]
        
        for ind in data.index: 
            smiles = data['SMILES'][ind]
            rdMol = Chem.MolFromSmiles(smiles) 
            nxMol = read_smiles(smiles, reinterpret_aromatic=True) 

            if ('.' in smiles):
                removed.append(ind)
            else:
                if nxMol.number_of_nodes() > 1:
                    if 'label' not in data.columns:
                        molinfo_list.append({
                        "Number": ind,
                        "SMILES": smiles,
                        "molobj": rdMol,
                        "molgraph": nxMol
                    })
                    else:
                        molinfo_list.append({
                            "Number": ind,
                            'class': data['label'][ind],
                            "SMILES": smiles,
                            "molobj": rdMol,
                            "molgraph": nxMol
                        })
                else:
                    removed.append(ind)

        
        molinfo_df = pd.DataFrame(molinfo_list, columns=columns)
        
        molinfo_df.index = molinfo_df["Number"]

        if len(removed) != 0:
            print(f'Molecules not allowed to Random Walks in Test Data:')
            for i in removed:
                smiles = data['SMILES'][i]
                print(f'\t{i}\t{smiles}')

        res_data = data[~data.index.isin(removed)]
        print(f'The shape of RW-allowed DILI molecules in Test Data: {res_data.shape}\n')

        return res_data, molinfo_df

    def prepare_rw_train(self, data): 
        
        molinfo_list = []
        removed    = []

        for diliid in data.index: 
            smiles = data['SMILES'][diliid]
            rdMol = Chem.MolFromSmiles(smiles) 
            nxMol = read_smiles(smiles, reinterpret_aromatic=True) 

            if ('.' in smiles):
                removed.append(diliid)
            else:
                if nxMol.number_of_nodes() > 1:

                    molinfo_list.append({
                        'ID': diliid,
                        'class': data['class'][diliid],
                        'SMILES': smiles,
                        'molobj': rdMol,
                        'molgraph': nxMol
                    })
                else:
                    removed.append(diliid)

        
        molinfo_df = pd.DataFrame(molinfo_list)
        molinfo_df.index = molinfo_df.ID

        if len(removed) != 0:
            print(f'Molecules not allowed to Random Walks in Training Data:')
            for i in removed:
                smiles = data['SMILES'][i]
                print(f'\t{i}\t{smiles}')

        res_data = data[~data.index.isin(removed)]
        print(f'The shape of RW-allowed DILI molecules in Training Data: {res_data.shape}\n')

        return res_data, molinfo_df
