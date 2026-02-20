import os
import pickle
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm


def count_patterns(smiles, patterns):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return [sum(1 for _ in mol.GetSubstructMatches(pattern)) for pattern in patterns]
    else:
        return [0] * len(patterns)


def process_dataframe(df, name, patterns, id_key, output_dir, threshold):
    tqdm.pandas()
    pattern_counts = df["SMILES"].progress_apply(lambda x: count_patterns(x, patterns))
    pattern_df = pd.DataFrame(pattern_counts.tolist(), columns=[f"P_{i+1}" for i in range(len(patterns))])
    result_df = pd.concat([df, pattern_df], axis=1)

    id2i = {v: i for i, v in enumerate(df[id_key])}
    subgraphFreq_matrix = np.zeros((len(id2i), len(patterns)), dtype=float)

    for i, row in tqdm(result_df.iterrows(), total=result_df.shape[0], desc=f"Building FP for {name}"):
        idx = id2i.get(row[id_key], None)
        if idx is not None:
            subgraphFreq_matrix[idx, :] = row.iloc[-len(patterns):].values

    out_path = output_dir / f"FP_{name}_{threshold}.pickle"
    pickle.dump({"subgraph": subgraphFreq_matrix}, open(out_path, "wb"))
    print(f"Saved: {out_path}")


def prepare_and_submine(args):
    raw_data_dir = Path(args.raw_data_dir).resolve()
    output_root = Path(args.output_root).resolve()
    output_dir = output_root / str(args.threshold) / args.target
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load DiSC patterns
    disc_df = pd.read_csv(args.disc_path)
    patterns = [Chem.MolFromSmarts(s) for s in disc_df["DiSC"].dropna()][:args.top]

    print(f"\n Processing target: {args.target}")

    # === Active ===
    active_path = raw_data_dir / args.target / "actives_final.ism"
    active_df = pd.read_csv(active_path, sep=r"\s+", header=None, names=["origin_SMILES", "CHEMBLID"])
    active_df.drop_duplicates(subset="origin_SMILES", keep='last', inplace=True)
    active_df['SMILES'] = active_df['origin_SMILES'].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    active_df.drop_duplicates(subset="SMILES", keep='last', inplace=True)
    active_df.drop_duplicates(subset='CHEMBLID', keep='last', inplace=True)
    active_df.sort_values(by='SMILES', ascending=False, inplace=True)
    active_df.reset_index(drop=True, inplace=True)
    active_df.to_csv(output_dir / "ACTIVE_DATA.csv", index=False)
    print(f"Saved: ACTIVE_DATA.csv")

    # === Decoy ===
    decoy_path = raw_data_dir / args.target / "decoys_final.ism"
    decoy_df = pd.read_csv(decoy_path, sep=r"\s+", header=None, names=["origin_SMILES", "DECOY_SRC"])
    decoy_df.drop_duplicates(subset="origin_SMILES", keep='last', inplace=True)
    decoy_df.drop_duplicates(subset="DECOY_SRC", keep='last', inplace=True)
    decoy_df['SMILES'] = decoy_df['origin_SMILES'].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    decoy_df.drop_duplicates(subset="SMILES", keep='last', inplace=True)
    decoy_df["DECOYID"] = decoy_df["DECOY_SRC"].apply(lambda x: f"D_{x}")
    decoy_df.sort_values(by="SMILES", ascending=False, inplace=True)
    decoy_df.reset_index(drop=True, inplace=True)
    decoy_df.to_csv(output_dir / "DECOY_DATA.csv", index=False)
    print(f"Saved: DECOY_DATA.csv")

    # === Seed ===
    seed_path = raw_data_dir / args.target / f"{args.target}_DATA_{args.threshold}.csv"
    seed_df = pd.read_csv(seed_path)
    seed_df = seed_df[seed_df["label"] == 1]
    seed_df.drop_duplicates(subset="SMILES", keep='last',inplace=True)
    seed_df.drop_duplicates(subset="cid", keep='last', inplace=True)
    seed_df.sort_values(by='SMILES', ascending=False, inplace=True)
    seed_df.reset_index(drop=True, inplace=True)
    seed_df.to_csv(output_dir / "SEED_DATA.csv", index=False)
    print(f"Saved: SEED_DATA.csv")

    print("Matching DiSC patterns...")
    process_dataframe(seed_df, "Seed", patterns, "cid", output_dir, args.threshold)
    process_dataframe(active_df, "Active", patterns, "CHEMBLID", output_dir, args.threshold)
    process_dataframe(decoy_df, "Decoy", patterns, "DECOYID", output_dir, args.threshold)

    print(f"Done: {args.target}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare DUDE data and apply DiSC substructure matching.")
    parser.add_argument("--target", type=str, required=True, help="Target name (e.g. ADA, EGFR)")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--top", type=int, default=100, help="Top N DiSC patterns")
    parser.add_argument("--disc_path", type=str, required=True, help="Path to DiSC pattern CSV")
    parser.add_argument("--raw_data_dir", type=str, default="./data/DUDE")
    parser.add_argument("--output_root", type=str, default="./SUBMINE/data/DUDE")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_and_submine(args)
