# ------------------------------------------------------------------------------------------------
# --- Sample creation for HBN (individual-level analysis) ---
# ------------------------------------------------------------------------------------------------

# Run after create_group_level_hbn.R. 
# Creates a final sample csv in data/HBN/derivatives/final_sample.
# ------------------------------------------------------------------------------------------------

import os
import pandas as pd

# ------------------------------------------------------------------------------------------------
# --- set up inputs ---
# ------------------------------------------------------------------------------------------------

root = "/Volumes/tractmaps"
data_root = f"{root}/data/HBN/derivatives"
cubic_root = f"{root}/data/HBN/behavioral_data"
output_dir = f"{data_root}/final_sample"

# General QC parameters
EXCLUDE_SITES = ["si"] # Site: exclude SI (1.5T)
NUM_DIRECTIONS_REQUIRED = 128
T1_NEIGHBOR_CORR_MIN = 0.9
MEAN_FD_MAX = 1.0
METRICS = ["fa", "md", "icvf"]

# ------------------------------------------------------------------------------------------------
# --- output folder ---
# ------------------------------------------------------------------------------------------------

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Folder '{output_dir}' created.")
else:
    print(f"Folder '{output_dir}' already exists.")

# ------------------------------------------------------------------------------------------------
# --- load data ---
# ------------------------------------------------------------------------------------------------

qc_path = os.path.join(data_root, "cleaned", "hbn_qc_measures.csv")
if not os.path.isfile(qc_path):
    raise FileNotFoundError(f"Run create_group_level_hbn.R first. Missing: {qc_path}")
dwi_qc = pd.read_csv(qc_path)
print(f"QC data: N = {len(dwi_qc)} rows")

participants_path = os.path.join(cubic_root, "participants.tsv")
if not os.path.isfile(participants_path):
    raise FileNotFoundError(f"Demographics file not found: {participants_path}")
demographics = pd.read_csv(participants_path, sep="\t")
print(f"Demographics data: N = {len(demographics)}")

# ------------------------------------------------------------------------------------------------
# --- helper ---
# ------------------------------------------------------------------------------------------------


def _subject_ids_complete_tracts(data_root_local: str, metrics: list):
    """
    Intersection of subject_id that have no NA in any tract column for each metric.
    """
    complete = None
    found_any = False
    for metric in metrics:
        tract_path = os.path.join(data_root_local, "cleaned", f"hbn_tracts_{metric}.csv")
        if not os.path.isfile(tract_path):
            continue
        tracts = pd.read_csv(tract_path)
        if "subject_id" not in tracts.columns:
            continue
        prefix = f"{str(metric).strip().lower()}_"
        val_cols = [c for c in tracts.columns if str(c).lower().startswith(prefix)]
        if not val_cols:
            continue
        found_any = True
        sid = tracts["subject_id"].astype(str).str.strip()
        mask = tracts[val_cols].notna().all(axis=1)
        ids_ok = set(sid[mask])
        complete = ids_ok if complete is None else complete & ids_ok
    if not found_any:
        return None
    return complete



# ------------------------------------------------------------------------------------------------
# --- demographics: subject_id, sex, age, site ---
# ------------------------------------------------------------------------------------------------

demographics = demographics.rename(
    columns={
        "participant_id": "subject_id",
        "Sex": "sex",
        "Age": "age",
    }
)
demographics["subject_id"] = demographics["subject_id"].astype(str).str.strip()
sex_map = {0: "male", 1: "female"}
if demographics["sex"].dtype in ("int64", "float64", "int32"):
    demographics["sex"] = demographics["sex"].map(sex_map)

keep_cols = ["subject_id", "sex", "age", "site"]
demographics = demographics[[c for c in keep_cols if c in demographics.columns]].copy()

initial_sample_size = len(demographics)
print(f"Initial sample size: N = {initial_sample_size}")

# ------------------------------------------------------------------------------------------------
# --- site exclusion (exclude SI) ---
# ------------------------------------------------------------------------------------------------

print(f"Applying site exclusion: exclude sites = {EXCLUDE_SITES}")
site_lower = demographics["site"].astype(str).str.strip().str.lower()
ex_lower = {s.lower() for s in EXCLUDE_SITES}
n_before_site = len(demographics)
demographics = demographics[~site_lower.isin(ex_lower)].copy()
print(f"After site exclusion: N = {len(demographics)}")
print(f"Number of participants excluded: {n_before_site - len(demographics)}")
# ------------------------------------------------------------------------------------------------
# --- non-variant diffusion exclusion ---
# ------------------------------------------------------------------------------------------------

print("Applying non-variant diffusion MRI exclusion.")
n_before_variant = len(demographics)
site_l = demographics["site"].astype(str).str.strip().str.lower()
is_non_variant = ~dwi_qc["acq_id"].astype(str).str.contains("VARIANT", na=False)
dwi_qc = dwi_qc[is_non_variant].copy()
keep_subjects = dwi_qc["subject_id"].unique()
demographics = demographics[demographics["subject_id"].isin(keep_subjects)].copy()

n_after_variant = len(demographics)
print(f"After filtering to non-variant DWI participants: N = {n_after_variant}")
print(
    "Number of participants excluded due to variant diffusion MRI data: "
    f"{n_before_variant - n_after_variant}"
)

# ------------------------------------------------------------------------------------------------
# --- diffusion acquisition exclusion ---
# ------------------------------------------------------------------------------------------------

n_before_acquisition = len(demographics)
subjects_num_dir = dwi_qc[dwi_qc["raw_num_directions"] == NUM_DIRECTIONS_REQUIRED][
    "subject_id"
].unique()
demographics = demographics[demographics["subject_id"].isin(subjects_num_dir)].copy()
n_after_acquisition = len(demographics)
print(
    f"After diffusion acquisition exclusion (raw_num_directions == {NUM_DIRECTIONS_REQUIRED}): "
    f"N = {n_after_acquisition}"
)
print(f"Number of participants excluded: {n_before_acquisition - n_after_acquisition}")

# ------------------------------------------------------------------------------------------------
# --- diffusion quality exclusion ---
# ------------------------------------------------------------------------------------------------

n_before_quality = len(demographics)
quality_exclude = dwi_qc[dwi_qc["t1_neighbor_corr"] < T1_NEIGHBOR_CORR_MIN][
    "subject_id"
].unique()
demographics = demographics[~demographics["subject_id"].isin(quality_exclude)].copy()
n_after_quality = len(demographics)
print(
    f"After diffusion quality exclusion (t1_neighbor_corr < {T1_NEIGHBOR_CORR_MIN}): "
    f"N = {n_after_quality}"
)
print(f"Number of participants excluded: {n_before_quality - n_after_quality}")

# ------------------------------------------------------------------------------------------------
# --- motion exclusion ---
# ------------------------------------------------------------------------------------------------

n_before_motion = len(demographics)
motion_exclude = dwi_qc[dwi_qc["mean_fd"] > MEAN_FD_MAX]["subject_id"].unique()
demographics = demographics[~demographics["subject_id"].isin(motion_exclude)].copy()
n_after_motion = len(demographics)
print(f"After motion exclusion (mean_fd > {MEAN_FD_MAX}): N = {n_after_motion}")
print(f"Number of participants excluded: {n_before_motion - n_after_motion}")

# ------------------------------------------------------------------------------------------------
# --- merge QC (mean_fd) ---
# ------------------------------------------------------------------------------------------------

qc_cols = [c for c in ["subject_id", "mean_fd"] if c in dwi_qc.columns]
dwi_qc_final = dwi_qc[qc_cols]
working = demographics.merge(dwi_qc_final, on="subject_id", how="left")

# ------------------------------------------------------------------------------------------------
# --- tract completeness (intersection across fa, md, icvf) ---
# ------------------------------------------------------------------------------------------------

tract_complete_ids = _subject_ids_complete_tracts(data_root, METRICS)
n_before_tract = len(working)
if tract_complete_ids is not None:
    sid = working["subject_id"].astype(str).str.strip()
    working = working[sid.isin(tract_complete_ids)].copy()
    print(
        "After tract completeness filter (intersection across metrics; no NA in tract columns): "
        f"N = {len(working)}"
    )
    print(f"Number of participants excluded: {n_before_tract - len(working)}")
else:
    print("Warning: no tract CSVs found under cleaned/; tract completeness filter skipped.")

# ------------------------------------------------------------------------------------------------
# --- NA filter (all columns) ---
# ------------------------------------------------------------------------------------------------

n_before_na = len(working)
final_sample = working.dropna(how="any").copy()
print(f"After NA filter (all columns): N = {len(final_sample)}")
print(f"Number of participants excluded: {n_before_na - len(final_sample)}")

# ------------------------------------------------------------------------------------------------
# --- descriptive statistics ---
# ------------------------------------------------------------------------------------------------

_a = final_sample
print(f"\nDescriptives — full sample: N = {len(_a)}")
if _a["age"].notna().any():
    print(f"Mean age: {_a['age'].mean():.1f} years")
    print(f"SD age: {_a['age'].std():.1f} years")
    print(f"Age range: {_a['age'].min():.1f} - {_a['age'].max():.1f} years")
print(f"Sex distribution: {_a['sex'].value_counts().to_dict()}")
if "site" in _a.columns:
    print(f"Site distribution: {_a['site'].value_counts().to_dict()}")
if "mean_fd" in _a.columns:
    print(f"Mean mean_fd: {_a['mean_fd'].mean():.2f}")
    print(f"SD mean_fd: {_a['mean_fd'].std():.2f}")

# ------------------------------------------------------------------------------------------------
# --- add tract scalars and save (full sample only) ---
# ------------------------------------------------------------------------------------------------

print(f"\nFinal sample: N = {len(final_sample)}")

for metric in METRICS:
    tract_path = os.path.join(data_root, "cleaned", f"hbn_tracts_{metric}.csv")
    if not os.path.isfile(tract_path):
        print(f"  Skip {metric}: missing {tract_path}")
        continue
    tracts = pd.read_csv(tract_path)
    if "subject_id" not in tracts.columns:
        print(f"  Skip {metric}: no subject_id in {tract_path}")
        continue
    sample_full = final_sample.merge(tracts, on="subject_id", how="inner")
    out_full = os.path.join(output_dir, f"hbn_final_sample_{metric}.csv")
    sample_full.to_csv(out_full, index=False)
    print(f"  {metric}: N={len(sample_full)} -> {out_full}")

print(f"\nFiles saved to: {output_dir}")
