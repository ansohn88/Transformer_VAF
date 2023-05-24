import pickle

import numpy as np
import pandas as pd

output_parent_dir = "/home/andy/baraslab/projects/germline_somatic/Data"
counts_pickle_file = f"{output_parent_dir}/counts.pkl"
annot_pickle_file = f"{output_parent_dir}/tcga.genie.combined.annot.maf.pkl"
# purity_file = f"{output_parent_dir}/gsc/TCGA_mastercalls.abs_tables_JSedit.fixed.txt"

count = pd.read_pickle(counts_pickle_file)
annot = pd.read_pickle(annot_pickle_file)
# PURITY
# purity = pd.read_csv(purity_file, sep="\t")

count = count.loc[
    (count["FILTER"] == "PASS")
    | (count["FILTER"] == "wga")
    | (count["FILTER"] == "native_wga_mix")
]

count["sample_id"] = count["SAMPLE"].apply(lambda x: x[:12])
# count['is_germline'] = (count['LINEAGE'] == 'germline').astype(int)
count["is_somatic"] = (count["LINEAGE"] == "somatic").astype(int)
count["vaf"] = count["alt_count"] / (count["ref_count"] + count["alt_count"])


# # PURITY
# purity["sample_id"] = purity["sample"].apply(lambda x: x[:12])
# purity = purity.loc[purity["call status"] == "called"]
# purity = purity.groupby("sample_id")["purity"].mean().reset_index()

# count = pd.merge(count, purity, how="inner", on=["sample_id"])
# count['purity'] = count['purity'].fillna(1.0)
# count["vaf_purity"] = count["vaf"] * count["purity"]

count = count.loc[
    (count["LINEAGE"] != "both")
    & (~count["assembly_error"])
    & (count[["ref_count", "alt_count"]].sum(axis=1) > 0)
]


# annot[0].columns

annot_fields = [
    "ID",
    "INFO/gnomad_exomes_AC",
    "INFO/gnomad_exomes_AF",
    "INFO/gnomad_exomes_AC_popmax",
    "INFO/gnomad_exomes_AF_popmax",
    "INFO/gnomad_genomes_AC",
    "INFO/gnomad_genomes_AF",
    "INFO/gnomad_genomes_AC_popmax",
    "INFO/gnomad_genomes_AF_popmax",
]

merged = pd.merge(count, annot[0][annot_fields], on="ID", how="left")

merged = merged.loc[
    (merged["LINEAGE"] != "both")
    & (~merged["assembly_error"])
    & (merged[["ref_count", "alt_count"]].sum(axis=1) > 0)
]

predictor_variables = [
    "alt_count",
    "vaf",
    # 'purity',
    # 'vaf_purity',
    "INFO/gnomad_exomes_AC",
    "INFO/gnomad_exomes_AF",
    "INFO/gnomad_exomes_AC_popmax",
    "INFO/gnomad_exomes_AF_popmax",
    "INFO/gnomad_genomes_AC",
    "INFO/gnomad_genomes_AF",
    "INFO/gnomad_genomes_AC_popmax",
    "INFO/gnomad_genomes_AF_popmax",
]

# # g = merged.groupby('bcr_patient_barcode')
g = merged.groupby("sample_id")
tcga_id, counts, labels = list(
    zip(
        *[
            [
                name,
                values[predictor_variables].values,
                values["is_somatic"].values[:, np.newaxis],
            ]
            for name, values in g
        ]
    )
)

save_dir = f"{output_parent_dir}/gsc"
pickle.dump(
    {"tcga_id": tcga_id, "counts": counts, "labels": labels},
    # open(f'{save_dir}/gsc_purity_data_dict.pkl',
    open(f"{save_dir}/gsc_data_dict.pkl", "wb"),
)
