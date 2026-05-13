# ------------------------------------------------------------------------------------------------
# --- Sample creation for HCP-YA (individual-level analysis) ---
# ------------------------------------------------------------------------------------------------

# Run after create_group_level_hcpya.R.
# Creates a final sample csv in data/HCPYA/derivatives/final_sample.
# ------------------------------------------------------------------------------------------------

import os
import pandas as pd

# ------------------------------------------------------------------------------------------------
# --- set up inputs ---
# ------------------------------------------------------------------------------------------------

root = "/Volumes/tractmaps"
data_root = f"{root}/data/HCPYA/derivatives"
cubic_root = f"{root}/data/HCPYA/behavioral_data"
demog_cognition_path = "HCP_YA_subjects_2026_04_10_13_58_31.csv"

# Group-level tract measures
tracts_fa_relpath = f"{data_root}/cleaned/hcpya_tracts_fa.csv"
tracts_md_relpath = f"{data_root}/cleaned/hcpya_tracts_md.csv"

# Cognition column in HCP_YA subject table
COGNITION_COL = "CogFluidComp_Unadj"
SCANNER_COL = "MRsession_Scanner_3T"  # we're keeping 3T sessions only

# Output directory
output_dir = f"{data_root}/final_sample"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Folder '{output_dir}' created.")
else:
    print(f"Folder '{output_dir}' already exists.")


# ------------------------------------------------------------------------------------------------
# --- helper ---
# ------------------------------------------------------------------------------------------------

def _hcpya_sub_prefix(x: pd.Series) -> pd.Series:
    """BIDS-style subject_id: add sub- if not already present."""
    s = x.astype(str).str.strip()
    return s.apply(lambda v: v if str(v).startswith("sub-") else f"sub-{v}")

# ------------------------------------------------------------------------------------------------
# --- load data ---
# ------------------------------------------------------------------------------------------------

# Demographics and cognition
demographics_path = os.path.join(cubic_root, demog_cognition_path)
if not os.path.isfile(demographics_path):
    raise FileNotFoundError(f"Demographics/cognition file not found: {demographics_path}")
demographics = pd.read_csv(demographics_path)
print(f"Demographics/cognition (raw) data: N = {len(demographics)}")

if not os.path.isfile(tracts_fa_relpath) or not os.path.isfile(tracts_md_relpath):
    raise FileNotFoundError(
        f"Tract group-level CSVs missing. Expected: {tracts_fa_relpath} and {tracts_md_relpath}"
    )
tracts_fa = pd.read_csv(tracts_fa_relpath)
tracts_md = pd.read_csv(tracts_md_relpath)
print("Loaded hcpya_tracts_fa and hcpya_tracts_md (group level).")
print(f"Cognition column used: {COGNITION_COL}")

# ------------------------------------------------------------------------------------------------
# --- Select 3T sessions only ---
# ------------------------------------------------------------------------------------------------

if SCANNER_COL not in demographics.columns:
    raise ValueError(
        f"HCP-YA subject table must include {SCANNER_COL!r} to restrict to 3T sessions."
    )
_n_pre = len(demographics)
demographics = demographics[
    demographics[SCANNER_COL].astype(str).str.strip() == "HCP3T"
].copy()
print(
    f"HCP-YA {SCANNER_COL} == 'HCP3T' → N = {len(demographics)} "
    f"(excluded {_n_pre - len(demographics)} rows)"
)

if "Subject" not in demographics.columns:
    raise ValueError('Missing id column: "Subject"')
if COGNITION_COL not in demographics.columns:
    raise ValueError(
        f"Missing cognition column: {COGNITION_COL!r} (revisions/6 hcpya CONFIG['cognition_col'])"
    )

# Age (HCP-YA subject table: Age_in_Yrs)
demographics["age"] = pd.to_numeric(demographics["Age_in_Yrs"], errors="coerce")

# Sex (HCP-YA subject table: Gender)
demographics["sex"] = demographics["Gender"]

# Subject ID
demographics["subject_id"] = _hcpya_sub_prefix(demographics["Subject"])

# Keep only necessary columns
keep = ["subject_id", "age", "sex", COGNITION_COL]
final_sample = demographics[[c for c in keep if c in demographics.columns]].copy()

print(f"Sample size (after 3T filter): N = {len(demographics)}")


# ------------------------------------------------------------------------------------------------
# --- merge tract scalars, then NA filters for covariates (FA) ---
# ------------------------------------------------------------------------------------------------

cognition_sample_fa = final_sample.merge(tracts_fa, on="subject_id", how="left")
n0 = len(cognition_sample_fa)

# Filter for complete cases in age, cognition, sex 
n0 = len(cognition_sample_fa)
meta_complete_cols = ["age", COGNITION_COL, "sex"]
missing_for_filter = [c for c in meta_complete_cols if c not in cognition_sample_fa.columns]
if missing_for_filter:
    raise ValueError(
        "After FA tract merge, expected columns for NA filter: "
        f"{meta_complete_cols}; missing: {missing_for_filter}"
    )
cognition_sample_fa = cognition_sample_fa.dropna(subset=meta_complete_cols, how="any")
print(
    f"Filter NA in {', '.join(meta_complete_cols)} → N = {len(cognition_sample_fa)} "
    f"(excluded {n0 - len(cognition_sample_fa)})"
)

# ------------------------------------------------------------------------------------------------
# --- merge tract scalars, then NA filters for covariates (MD) ---
# ------------------------------------------------------------------------------------------------

cognition_sample_md = final_sample.merge(tracts_md, on="subject_id", how="left")
n0 = len(cognition_sample_md)

# Filter for complete cases in age, cognition, sex 
n0 = len(cognition_sample_md)
missing_for_filter = [c for c in meta_complete_cols if c not in cognition_sample_md.columns]
if missing_for_filter:
    raise ValueError(
        "After MD tract merge, expected columns for NA filter: "
        f"{meta_complete_cols}; missing: {missing_for_filter}"
    )
cognition_sample_md = cognition_sample_md.dropna(subset=meta_complete_cols, how="any")
print(
    f"Filter NA in {', '.join(meta_complete_cols)} → N = {len(cognition_sample_md)} "
    f"(excluded {n0 - len(cognition_sample_md)})"
)

# ------------------------------------------------------------------------------------------------
# --- descriptive statistics ---
# ------------------------------------------------------------------------------------------------

_a = cognition_sample_fa
print(f"\nDescriptives — FA cognition sample: N = {len(_a)}")
print(f"Mean age: {_a['age'].mean():.1f} years")
print(f"SD age: {_a['age'].std():.1f} years")
print(
    f"Age range: {_a['age'].min():.1f} - {_a['age'].max():.1f} years"
)
print(f"Sex distribution: {_a['sex'].value_counts().to_dict()}")

# ------------------------------------------------------------------------------------------------
# --- save ---
# ------------------------------------------------------------------------------------------------

cognition_sample_fa.to_csv(
    os.path.join(output_dir, "hcpya_final_cognition_sample_fa.csv"), index=False
)
print(
    f"\nSaved: hcpya_final_cognition_sample_fa.csv (N = {len(cognition_sample_fa)})"
)

cognition_sample_md.to_csv(
    os.path.join(output_dir, "hcpya_final_cognition_sample_md.csv"), index=False
)
print(
    f"Saved: hcpya_final_cognition_sample_md.csv (N = {len(cognition_sample_md)})"
)
print(f"Output directory: {output_dir}")
