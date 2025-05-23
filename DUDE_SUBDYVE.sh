#!/bin/bash

TARGETS=(
    "ACES"
    "ADA"
    "ANDR"
    "EGFR"
    "FA10"
    "KIT"
    "PLK1"
    "SRC"
    "THRB"
    "UROK"
)

THRESHOLD=0.9
TOP=${1:-2000}
SIMILARITY="cosine"
EDGE_CUTOFF='0.85'
SEED_CUTOFF='0.5'

for target in "${TARGETS[@]}"
do
    echo "Step 1: Matching DiSC FP for $target"

    python3 ./SUBMINE/bin/matchFP.py \
        --target "$target" \
        --threshold "$THRESHOLD" \
        --top "$TOP" \
        --disc_path "./SSM/results/DUDE/${THRESHOLD}/${target}/CONCAT_${target}_DISC_${TOP}_filtered.csv" \
        --raw_data_dir "./data/DUDE" \
        --output_root "./SUBMINE/data/DUDE"

    echo "Step 2: Building network for $target"

    INPUT_PATH="./SUBMINE/data/DUDE/${THRESHOLD}/${target}"
    OUTPUT_PATH="./Network/DUDE/${target}/output_${THRESHOLD}"
    mkdir -p "${OUTPUT_PATH}"

    ACTIVE_FILE="${INPUT_PATH}/ACTIVE_DATA.csv"
    SEED_FILE="${INPUT_PATH}/SEED_DATA.csv"
    DECOY_FILE="${INPUT_PATH}/DECOY_DATA.csv"

    python3 source/makeNetwork_combined.py \
        --active "${ACTIVE_FILE}" \
        --seed "${SEED_FILE}" \
        --decoy "${DECOY_FILE}" \
        --input_fp_dir "${INPUT_PATH}" \
        --output "${OUTPUT_PATH}" \
        --compound_cutoff "${THRESHOLD}" \
        --seed_cutoff "${SEED_CUTOFF}" \
        --edge_cutoff "${EDGE_CUTOFF}" \
        --similarity "${SIMILARITY}"

    echo "Step 3: Running network propagation for $target"

    FP_ACTIVE="${INPUT_PATH}/FP_Active_${THRESHOLD}.pickle"
    FP_SEED="${INPUT_PATH}/FP_Seed_${THRESHOLD}.pickle"
    FP_DECOY="${INPUT_PATH}/FP_Decoy_${THRESHOLD}.pickle"

    PT_CB_ACTIVE="${INPUT_PATH}/DeepChem_ChemBERTa-77M-MTR_${target}_Active.pkl"
    PT_CB_SEED="${INPUT_PATH}/DeepChem_ChemBERTa-77M-MTR_${target}_Seed.pkl"
    PT_CB_DECOY="${INPUT_PATH}/DeepChem_ChemBERTa-77M-MTR_${target}_Decoy.pkl"

    sNPSeed="${OUTPUT_PATH}/seed_${SEED_CUTOFF}.txt"
    sNPEdge=$(find "${OUTPUT_PATH}" -name "edge_${SEED_CUTOFF}_${EDGE_CUTOFF}_"* | head -n 1)

    if [[ -z "$sNPEdge" ]]; then
        echo "No edge file found for ${target}, skipping..."
        continue
    fi

    sNPresult=$(echo "$sNPEdge" | sed -e "s/edge/NP_SUBDYVE/")

    python3 source/DUDE_NP_SUBDYVE.py \
        "$sNPEdge" "$sNPSeed" \
        -o "$sNPresult" \
        -addBidirectionEdge True \
        -data_SEED "$SEED_FILE" -data_ACTIVE "$ACTIVE_FILE" -data_DECOY "$DECOY_FILE" \
        -fp_SEED "$FP_SEED" -fp_ACTIVE "$FP_ACTIVE" -fp_DECOY "$FP_DECOY" \
        -PT_CB_SEED "$PT_CB_SEED" -PT_CB_ACTIVE "$PT_CB_ACTIVE" -PT_CB_DECOY "$PT_CB_DECOY"

    echo "Step 4: Exclude seeds for $target"

    python3 ./source/utils.py \
        --target "$target" \
        --threshold "$THRESHOLD" \
        --seed_cutoff "$SEED_CUTOFF" \
        --network_output_dir "$OUTPUT_PATH" \
        --submine_data_dir "./SUBMINE/data/DUDE" \
        --edge_cutoff "$EDGE_CUTOFF" 

    echo "Completed all steps for $target"
    echo "---------------------------------------------"
done

echo "All targets processed"
