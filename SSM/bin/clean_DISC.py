import argparse
import glob
import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from rdkit import Chem

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, required=True, help='Target protein name (e.g., ACES)')
parser.add_argument('--top', type=int, default=100, help='Top N entries to save')
parser.add_argument('--threshold', type=float, required=True, help='Threshold value used in directory names')
parser.add_argument('--log_dir', type=str, required=True, help='Path to the log directory containing AUC CSV')
parser.add_argument('--result_root', type=str, required=True, help='Root path to results/ directory')
args = parser.parse_args()

target = args.target
top = args.top
THRESHOLD = args.threshold
log_path = args.log_dir
result_root = args.result_root

seed_max_AUC_dict = pd.read_csv(f'{log_path}/target_max_AUC_dict_{THRESHOLD}.csv')

if target not in seed_max_AUC_dict['target'].values:
    raise ValueError(f"[ERROR] Target {target} not found in AUC dictionary.")

max_AUC_iteration = seed_max_AUC_dict[seed_max_AUC_dict['target'] == target]['max_AUC_iteration'].values[0]
print(f"Target: {target}, max_AUC_iteration: {max_AUC_iteration}")
print(f"Number of TOP : {top}")

def canonicalize_smarts(smarts):
    fragments = smarts.split('.')
    mols = [Chem.MolFromSmarts(frag) for frag in fragments if Chem.MolFromSmarts(frag)]
    if not mols:
        return None
    combined = mols[0]
    for mol in mols[1:]:
        combined = Chem.CombineMols(combined, mol)
    try:
        combined_mol = Chem.Mol(combined)
        Chem.SanitizeMol(combined_mol)
        return Chem.MolToSmiles(combined_mol, canonical=True)
    except Exception as e:
        print(f"Canonicalization failed: {smarts}\n→ {e}")
        return None

def remove_structurally_duplicate_smarts(df, smarts_col='DiSC'):
    unique_indices = []
    seen_canonical = set()
    for i, smarts in tqdm(enumerate(df[smarts_col]), total=len(df)):
        canonical = canonicalize_smarts(smarts)
        if canonical is None:
            continue
        if canonical not in seen_canonical:
            unique_indices.append(i)
            seen_canonical.add(canonical)
    return df.iloc[unique_indices].reset_index(drop=True)

result_path = f'{result_root}/{THRESHOLD}/{target}/'
DiSC_paths = glob.glob(f'{result_path}iteration_{max_AUC_iteration}/DiSC_*')

concat_df = pd.concat([pd.read_csv(path) for path in DiSC_paths], axis=0)
concat_df.sort_values(by='significance', ascending=False, inplace=True)
concat_df['DiSC_canonical'] = concat_df['DiSC'].apply(canonicalize_smarts)
concat_df_unique = concat_df.drop_duplicates(subset=["DiSC_canonical"], keep='first')
concat_df_deduped = remove_structurally_duplicate_smarts(concat_df_unique)
filtered_df = concat_df_deduped[concat_df_deduped['support F_1'] > concat_df_deduped['support F_0']].reset_index(drop=True)

output_file = f"{result_path}CONCAT_{target}_DISC_{top}_filtered.csv"
filtered_df.head(top).to_csv(output_file, index=False)
print(f"Saved: {output_file}")
