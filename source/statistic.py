import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as stats
import networkx as nx
from rdkit import Chem
from rdkit import DataStructs
import numba as nb
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Pool, cpu_count
import os, pickle

@nb.njit
def convertNPtoNum(NP):
    nSize = len(NP)
    nCutNum = math.ceil(nSize/64)
    arrNum = np.array([0]*nCutNum, dtype=np.uint64)

    for i in range(nCutNum):
        nSIdx = i * 64
        for j in range(64):
            nCurrIdx = nSIdx + j
            if nCurrIdx == nSize:   break
            if NP[nCurrIdx] == 1:   arrNum[i] |= (0b1 << (64-j-1))
    return arrNum

@nb.njit
def calcTanimoto(params):
    arrNum1, arrNum2 = params
    if len(arrNum1) != len(arrNum2):    return -1
    nCntInter, nCntUnion = 0, 0
    for i in range(len(arrNum1)):
        nN1, nN2 = arrNum1[i], arrNum2[i]
        nInter = (nN1 & nN2)
        nUnion = (nN1 | nN2)
        for j in range(64):
            nTest = (0b1 << j)
            if (nTest & nInter) == nTest:   nCntInter += 1
            if (nTest & nUnion) == nTest:   nCntUnion += 1
    dSim = 0
    if nCntInter != 0 and nCntUnion != 0:   dSim = nCntInter/nCntUnion
    return dSim

def CalcEuclideanForTwoFPs(listFP1, listFP2, dCutoff):
    nListFP1 = []
    nListFP2 = []
    listResult = []

    for i in range(len(listFP1)):
        for j in range(len(listFP2)):
            # Calculate Euclidean distance
            distance = np.linalg.norm(np.array(listFP1[i]) - np.array(listFP2[j]))
            distance = distance / 10 # Normalize Max value 
            distance = 1/(1+distance) # Convert to Low value is High similarity
            
            if distance >= dCutoff:
                listResult.append(f'{i}\t{j}\t{round(distance, 4)}')
    
    return listResult


def CalcTanimotoForTwoFPs(listFP1, listFP2, dCutoff):
    nListFP1 = []
    nListFP2 = []
    for itemFP in listFP1:
        nFP = 0
        nNum = len(itemFP)
        for i in range(nNum):     
            if itemFP[i] == 1:    nFP |= (0b1 << (nNum-i-1))
        nListFP1.append(nFP)
    for itemFP in listFP2:
        nFP = 0
        nNum = len(itemFP)
        for i in range(nNum):     
            if itemFP[i] == 1:    nFP |= (0b1 << (nNum-i-1))
        nListFP2.append(nFP)

    listResult = []
    for i in range(len(nListFP1)):
        for j in range(len(nListFP2)):
            nIntersection = bin((nListFP1[i] & nListFP2[j])).count("1")
            nUnion = bin((nListFP1[i] | nListFP2[j])).count("1")
            dSim = 0
            if nUnion != 0:         dSim = nIntersection/nUnion
            if dSim >= dCutoff:     
                listResult.append(str(i) + "\t" + str(j) + "\t" + str(round(dSim,4)))

    return listResult

def CalcTanimotoForTwoFPs_fast(listFP1, listFP2, dCutoff):
    X = np.array(listFP1, dtype=bool)  # shape: [n1, d]
    Y = np.array(listFP2, dtype=bool)  # shape: [n2, d]

    inter = np.dot(X, Y.T).astype(np.float32)  # [n1, n2]
    X_sum = X.sum(axis=1).reshape(-1, 1)        # [n1, 1]
    Y_sum = Y.sum(axis=1).reshape(1, -1)        # [1, n2]
    union = X_sum + Y_sum - inter               # [n1, n2]

    sim = np.divide(inter, union, out=np.zeros_like(inter), where=union != 0)
    idx_i, idx_j = np.where(sim >= dCutoff)

    return [
        f"{i}\t{j}\t{round(sim[i, j], 4)}"
        for i, j in zip(idx_i, idx_j)
    ]

def CalcTanimotoForTwoFPs_fast2(listFP1, listFP2, dCutoff, batch_size=10000):

    X = np.array(listFP1, dtype=bool)  # shape: (n1, dim)
    Y = np.array(listFP2, dtype=bool)  # shape: (n2, dim)
    n1, n2 = X.shape[0], Y.shape[0]
    
    bc1 = X.sum(axis=1)
    bc2 = Y.sum(axis=1)
    
    results = []
    for i in range(n1):
        row = X[i]
        cnt_i = bc1[i]
        
        for j0 in range(0, n2, batch_size):
            j1 = min(j0 + batch_size, n2)
            block = Y[j0:j1] 
            
            inter = np.bitwise_and(block, row).sum(axis=1)
            union = cnt_i + bc2[j0:j1] - inter
            
            sim = np.zeros_like(inter, dtype=float)
            nz = union > 0
            sim[nz] = inter[nz] / union[nz]
            
            mask = sim >= dCutoff
            for idx in np.nonzero(mask)[0]:
                j = j0 + idx
                results.append(f"{i}\t{j}\t{sim[idx]:.4f}")
                
    return results


def ParallelCalcTanimotoForTwoFPs(listParam):
    listCompFP, listSeedFP, dCutoff, sType, similarity = listParam
    print(sType, len(listCompFP), len(listSeedFP), dCutoff, similarity)
    if similarity == 'tanimoto':
        listResult = CalcTanimotoForTwoFPs_fast2(listCompFP, listSeedFP, dCutoff)
        
    elif similarity == 'cosine':
        listResult = CalcCosineSimilarityForTwoVectors(listCompFP, listSeedFP, dCutoff)
    elif similarity == 'euclidean':
        listResult = CalcEuclideanForTwoFPs(listCompFP, listSeedFP, dCutoff)
    else:
        raise ValueError("Unsupported similarity type: " + similarity)
    
    return listResult



def ParallelCalcTanimotoForFPs(listParam):
    listFP, dCutoff, sType, similarity = listParam
    print(sType, len(listFP), dCutoff, similarity)
    
    if similarity == 'tanimoto':
        listResult = CalcTanimotoForFPs_fast2(listFP, dCutoff)
    elif similarity == 'cosine':
        listResult = CalcCosineSimilarityForVectors(listFP, dCutoff)
    elif similarity == 'euclidean':
        listResult = CalcEuclideanForFPs(listFP, dCutoff)
    else:
        raise ValueError("Unsupported similarity type: " + similarity)
    return listResult

def CalcTanimotoForFPs(listFP, dCutoff, batch_size=10000000):
    nListFP = np.array(listFP, dtype=bool)
    nSize = len(nListFP)
    listResult = []

    for start_i in range(nSize):
        fp_i = nListFP[start_i]
        for start_j in range(start_i + 1, nSize, batch_size):
            end_j = min(start_j + batch_size, nSize)
            fp_batch = nListFP[start_j:end_j]

            intersection = np.logical_and(fp_i, fp_batch).sum(axis=1)
            union = np.logical_or(fp_i, fp_batch).sum(axis=1)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                similarity = np.true_divide(intersection, union)
                similarity[np.isnan(similarity)] = 0  
            
            mask = similarity >= dCutoff
            for idx, sim in zip(np.where(mask)[0], similarity[mask]):
                i = start_i
                j = start_j + idx
                listResult.append(f"{i}\t{j}\t{round(sim, 4)}")

    return listResult

def CalcTanimotoForFPs_fast(listFP, dCutoff):
    X = np.array(listFP, dtype=bool)
    inter = np.dot(X, X.T).astype(np.float32)
    bit_count = X.sum(axis=1)
    union = bit_count[:, None] + bit_count[None, :] - inter

    sim = np.divide(inter, union, out=np.zeros_like(inter), where=union != 0)
    idx_i, idx_j = np.triu_indices_from(sim, k=1)
    sim_vals = sim[idx_i, idx_j]

    mask = sim_vals >= dCutoff
    
    return [
        f"{i}\t{j}\t{round(s, 4)}"
        for i, j, s in zip(idx_i[mask], idx_j[mask], sim_vals[mask])
    ]
    
import numpy as np

def CalcTanimotoForFPs_fast2(listFP, dCutoff, batch_size=10000):
    X = np.array(listFP, dtype=bool)
    n = X.shape[0]
    bit_counts = X.sum(axis=1)
    results = []
    for i in range(n):
        row = X[i]
        cnt_i = bit_counts[i]
        for j0 in range(i + 1, n, batch_size):
            j1 = min(j0 + batch_size, n)
            block = X[j0:j1]                  
            inter = np.bitwise_and(block, row).sum(axis=1)
            union = cnt_i + bit_counts[j0:j1] - inter
            sim = np.zeros_like(inter, dtype=float)
            nz = union > 0
            sim[nz] = inter[nz] / union[nz]
            
            mask_idxs = np.nonzero(sim >= dCutoff)[0]
            for idx in mask_idxs:
                j = j0 + idx
                results.append(f"{i}\t{j}\t{sim[idx]:.4f}")

    return results

# Euclidean similarity
def CalcEuclideanForFPs(listFP, dCutoff):
    listResult = []
    nSize = len(listFP)
    
    for i in range(nSize):
        for j in range(i + 1, nSize):
            dSim = np.linalg.norm(np.array(listFP[i]) - np.array(listFP[j]))
            dSim = dSim / 10 # Normalize Max value 
            dSim = 1/(1+dSim) # Convert to Low value is High similarity
            if dSim >= dCutoff:
                listResult.append(f"{i}\t{j}\t{round(dSim, 4)}")

    return listResult

def CalcCosineSimilarityForVectors(listVectors, dCutoff):
    nListVectors = np.array(listVectors)
    cosine_sim_matrix = cosine_similarity(nListVectors)
    idx_i, idx_j = np.triu_indices_from(cosine_sim_matrix, k=1)
    similarities = cosine_sim_matrix[idx_i, idx_j]
    mask = similarities >= dCutoff
    listResult = [
        f"{i}\t{j}\t{round(sim, 4)}"
        for i, j, sim in zip(idx_i[mask], idx_j[mask], similarities[mask])
    ]
    return listResult

def ParallelCalcCosineSimilarityForVectors(listParam):
    listVectors, dCutoff, sType = listParam
    print(sType, len(listVectors), dCutoff)
    listResult = CalcCosineSimilarityForVectors(listVectors, dCutoff)
    return listResult

def CalcCosineSimilarityForTwoVectors(listFP1, listFP2, dCutoff):
    arrFP1 = np.array(listFP1)
    arrFP2 = np.array(listFP2)
    similarity_matrix = cosine_similarity(arrFP1, arrFP2)
    
    listResult = []
    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix[i])):
            similarity = similarity_matrix[i][j]
            if similarity >= dCutoff:
                listResult.append(str(i) + "\t" + str(j) + "\t" + str(round(similarity, 4)))
    
    return listResult

def ParallelCalcCosineSimilarityForTwoVectors(listParam):
    listCompFP, listSeedFP, dCutoff, sType = listParam
    print(sType, len(listCompFP), len(listSeedFP), dCutoff)
    listResult = CalcCosineSimilarityForTwoVectors(listSeedFP, listCompFP, dCutoff)
    return listResult

def removeRedundancy(active_path, seed_path, decoy_path):
    dfActive = pd.read_csv(active_path)
    dfSeed = pd.read_csv(seed_path)
    dfDecoy = pd.read_csv(decoy_path)
    
    dfActive.drop_duplicates(subset=['SMILES'], keep='last', inplace=True)
    dfActive.sort_values(by='SMILES', inplace=True)
    dfActive.rename(columns={'SMILES': 'smiles', 'CHEMBLID': 'id'}, inplace=True)
    dfActive.reset_index(drop=True, inplace=True)

    dfSeed.drop_duplicates(subset=['SMILES'], keep='last', inplace=True)
    dfSeed.sort_values(by='SMILES', inplace=True)
    dfSeed = dfSeed[dfSeed['activityid'] == 'Active']
    dfSeed.rename(columns={'SMILES': 'smiles'}, inplace=True)
    dfSeed["id"] = ""
    dfSeed = dfSeed.reset_index(drop=True)
    dfSeed['index'] = dfSeed.index.astype(str)
    dfSeed['acvalue'] = dfSeed['acvalue'].astype(str)
    dfSeed['id'] = 't' + dfSeed['index'] + '_' + dfSeed['acvalue']
    dfSeed['acvalue'] = dfSeed['acvalue'].astype(float)
    dfSeed.drop(['index'], axis=1, inplace=True)

    dfDecoy.drop_duplicates(subset=['SMILES'], keep='last', inplace=True)
    dfDecoy.sort_values(by='SMILES', inplace=True)
    dfDecoy.rename(columns={'SMILES': 'smiles', 'DECOYID': 'id'}, inplace=True)
    dfDecoy.reset_index(drop=True, inplace=True)

    return dfActive, dfSeed, dfDecoy

def load_existing_fps(input_path, cutoff_score):
    def load_fp(label):
        file_path = os.path.join(input_path, f"FP_{label}_{cutoff_score}.pickle")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"FP file not found: {file_path}")
        return pickle.load(open(file_path, "rb"))

    return {
        "active": load_fp("Active"),
        "seed": load_fp("Seed"),
        "decoy": load_fp("Decoy"),
    }
