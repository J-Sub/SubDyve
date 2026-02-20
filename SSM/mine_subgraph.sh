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
TOP=${1:-2000}  # default 2000

echo "Step 1: Running SSM on all targets..."

# Step 1: SSM
for target in "${TARGETS[@]}"; do
    echo "Running SSM for target: $target"

    train_data="./SSM/data/DUDE/${target}/train_data_${THRESHOLD}.csv"
    test_data="./SSM/data/DUDE/${target}/test_data_${THRESHOLD}.csv"
    output_dir="./SSM/results/DUDE/${THRESHOLD}/${target}/"
    mkdir -p "$output_dir"

    python ./SSM/bin/ssm_smiles.py \
        --train_data "$train_data" \
        --test_data "$test_data" \
        --output_dir "$output_dir" \
        --rw 7 --alpha 0.1 --iterations 20 --nWalker 5
done

echo "Step 1 completed."

# Step 2: best iteration collect
echo "Step 2: Aggregating best AUC iterations..."
python3 ./SSM/bin/collect.py --threshold "$THRESHOLD" --output_path "./SSM/logs/target_max_AUC_dict_${THRESHOLD}.csv"
echo "Step 2 completed."

# Step 3: Run DiSC-aware SSM using best iteration
echo "Step 3: Running DiSC refinement with best iterations..."
seed_max_AUC_dict="./SSM/logs/target_max_AUC_dict_${THRESHOLD}.csv"

for target in "${TARGETS[@]}"; do
    echo "Running DiSC refinement for: $target"
    
    bestIteration=$(awk -F, -v seed="$target" '$1 == seed {print $2}' ${seed_max_AUC_dict})

    train_data="./SSM/data/DUDE/${target}/train_data_${THRESHOLD}.csv"
    test_data="./SSM/data/DUDE/${target}/test_data_${THRESHOLD}.csv"
    output_dir="./SSM/results/DUDE/${THRESHOLD}/${target}/"
    mkdir -p "$output_dir"

    python ./SSM/bin/ssm_DISC.py \
        --train_data "$train_data" \
        --test_data "$test_data" \
        --output_dir "$output_dir" \
        --rw 7 --alpha 0.1 --iterations 20 \
        --nWalker 5 --DiSC 3 \
        --bestIteration "$bestIteration"
done

echo "Step 3 completed."

# Step 4: Extract subgraphs
echo "Step 4: Removing DISC top-$TOP from prediction results..."

for target in "${TARGETS[@]}"; do
    echo "Processing DISC removal for: $target"
    python3 ./SSM/bin/clean_DISC.py \
        --target "$target" \
        --top "$TOP" \
        --threshold "$THRESHOLD" \
        --log_dir "./SSM/logs" \
        --result_root "./SSM/results/DUDE"
done

echo "All TARGETS processed."
