#!/bin/bash

RANDOM_SEEDS=(0 1 2 3 4)
# RANDOM_SEEDS=(0)

for RANDOM_SEED in "${RANDOM_SEEDS[@]}"; do
    dCompoundCutoff=0.0
    dSeedCutoff=500
    dEdgeCutoff=0.85

    basePath="./exp_PU/exp_${RANDOM_SEED}"
    sPath="${basePath}/Network/"
    mkdir -p "${sPath}"
    outputDir="${sPath}output/"
    mkdir -p "${outputDir}"

    DATA_SEED="${basePath}/Data_Seed_${dCompoundCutoff}.csv"
    DATA_COMP="${basePath}/Data_Comp_${dCompoundCutoff}.csv"
    FP_SEED="${basePath}/FP_Seed_${dCompoundCutoff}.pickle"
    FP_COMP="${basePath}/FP_Comp_${dCompoundCutoff}.pickle"
    PT_CB_SEED="${basePath}/DeepChem_ChemBERTa-77M-MTR_${RANDOM_SEED}_seed.pkl"
    PT_CB_COMP="${basePath}/DeepChem_ChemBERTa-77M-MTR_${RANDOM_SEED}_comp.pkl"
    sSelectedIDPath="${basePath}/selected_ids_${RANDOM_SEED}.txt"
    sNPSeed="${basePath}/seed_${RANDOM_SEED}.txt"
    NPresult="NP_SUBDYVE_${dSeedCutoff}_${dEdgeCutoff}_subgraph.txt"

    sNPEdge="${basePath}/edge_${dSeedCutoff}_${dEdgeCutoff}_subgraph.txt"
    sNPresult="${sNPEdge/edge/NP_SUBDYVE}"

    python3 source/PU_NP_SUBDYVE.py \
        "${sNPEdge}" "${sNPSeed}" \
        -o "${sNPresult}" \
        -addBidirectionEdge True \
        -data_seed "${DATA_SEED}" -data_comp "${DATA_COMP}" \
        -fp_seed "${FP_SEED}" -fp_comp "${FP_COMP}" \
        -PT_CB_seed "${PT_CB_SEED}" -PT_CB_comp "${PT_CB_COMP}"


    python3 - <<END # exclude seed
import pandas as pd

selected_ids = pd.read_csv('${sSelectedIDPath}', sep='\t', header=None)[0].tolist()
NPresult = pd.read_csv('${basePath}/${NPresult}', sep='\t', header=None, names=['id', 'npscore'])
NPresult['id'] = NPresult['id'].apply(lambda x: '*' + x if x in selected_ids else x)
NPresult.sort_values(by='npscore', ascending=False, inplace=True)
NPresult.reset_index(drop=True, inplace=True)

seed_ids = pd.read_csv('${sNPSeed}', sep='\t', header=None)[0].tolist()
NPresult_no_seed = NPresult[~NPresult['id'].isin(seed_ids)]
NPresult_no_seed.to_csv('${outputDir}NP_SUBDYVE_result_excludeseed_${RANDOM_SEED}.txt', sep='\t', index=False)
END
done
