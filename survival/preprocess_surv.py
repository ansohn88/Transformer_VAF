import pickle

import numpy as np
import pandas as pd

CANCER_TYPES = [
    "BLCA",
    "CESC",
    "COAD",
    "ESCA",
    "HNSC",
    "KIRP",
    "LUAD",
    "LUSC",
    "OV",
    "PAAD",
    "SARC",
    "STAD",
    "UCEC",
]

EXCEL_COLS = [
    "bcr_patient_barcode",
    "type",
    "vital_status",
    "OS.time",
    "DSS.time",
    "DFI.time",
    "PFI.time",
]

DROPNA = "OS.time"


def get_vital_status(df: pd.DataFrame, alive: bool) -> pd.DataFrame:
    if alive:
        status_df = df[df["vital_status"] == "Alive"]
    else:
        status_df = df[df["vital_status"] == "Dead"]
    return status_df


def subset_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df.loc[:, columns]


def subset_rows(df: pd.DataFrame, row_indices: list[int]) -> pd.DataFrame:
    return df.iloc[row_indices, :]


def save_clinical_subset_as_pkl(
    in_dir: str = "/home/asohn3/baraslab/germline_somatic/Data/survival",
):
    clin_fu_file = f"{in_dir}/clinical_PANCAN_patient_with_followup.tsv"

    SUBSET_COLS = [
        "bcr_patient_barcode",
        "acronym",
        "histological_type",
        "vital_status",
        "days_to_death",
        "days_to_last_followup",
        "days_to_last_known_alive",
    ]

    followup = pd.read_csv(
        clin_fu_file, sep="\t", encoding="windows-1254", low_memory=False
    )

    subset_followup = subset_columns(df=followup, columns=SUBSET_COLS)
    subset_followup = subset_followup[
        subset_followup["vital_status"] != "[Not Available]"
    ]
    subset_followup = subset_followup[
        subset_followup["vital_status"] != "[Discrepancy]"
    ]

    subset_followup.to_pickle(
        "/home/asohn3/baraslab/germline_somatic/Data/survival/survival_labels.pkl"
    )

    return subset_followup


def merge_maf_surv():
    # surv = save_clinical_subset_as_pkl()

    gnomad, counts = pd.read_pickle(
        "/home/asohn3/baraslab/germline_somatic/Data/tcga.genie.combined.annot.maf.pkl"
    )
    surv_lbls = pd.read_pickle(
        "/home/asohn3/baraslab/germline_somatic/Data/survival/survival_labels.pkl"
    )

    counts["bcr_patient_barcode"] = counts["SAMPLE"].apply(lambda x: x[:12])
    merged = pd.merge(counts, surv_lbls, how="left", on=["bcr_patient_barcode"])
    merged_subset = merged.loc[merged["acronym"].isin(CANCER_TYPES)]

    merged.to_pickle(
        "/home/asohn3/baraslab/germline_somatic/Data/survival/counts_survival.pkl"
    )
    merged_subset.to_pickle(
        "/home/asohn3/baraslab/germline_somatic/Data/survival/rec_counts_survival.pkl"
    )


class SurvivalDataSetup:
    def __init__(
        self,
        original_counts_file: str,
        tcga_surv_excel_file: str,
        maf_file: str,
        survival_mtlr_lbls: str,
        survival_subset: bool,
        out_dir: str = "/home/asohn3/baraslab/germline_somatic/Data/final_surv",
    ) -> None:
        self.og_counts = pd.read_pickle(original_counts_file)

        self.gnomad, _ = pd.read_pickle(maf_file)
        self.tcga_surv_excel = pd.read_excel(
            tcga_surv_excel_file, sheet_name="TCGA-CDR", index_col=0
        )
        self.surv_mtlr_lbls = pd.read_pickle(survival_mtlr_lbls)
        if survival_subset:
            self.tcga_surv_excel = self.tcga_surv_excel.loc[
                self.tcga_surv_excel["type"].isin(CANCER_TYPES)
            ]

        self.out_dir = out_dir

    def apply_init(self, save_final_df: bool = False):
        self.og_counts = self.og_counts.drop(
            labels=[
                "acronym",
                "vital_status",
                "days_to_death",
                "days_to_last_followup",
                "histological_type",
            ],
            axis=1,
        )
        self.tcga_surv_excel = self.tcga_surv_excel[EXCEL_COLS]
        self.tcga_surv_excel["event"] = self.tcga_surv_excel["vital_status"].apply(
            lambda x: 1 if x == "Dead" else 0
        )
        self.merged_df = pd.merge(
            left=self.og_counts,
            right=self.tcga_id,
            how="left",
            on="bcr_patient_barcode",
        )
        self.merged_df = pd.merge(
            left=self.merged_df, right=self.gnomad, how="left", on="ID"
        )
        self.merged_df = self.merged_df.dropna(subset=DROPNA)

        if save_final_df:
            self.merged_df.to_pickle(
                "/home/asohn3/baraslab/germline_somatic/Data/final_surv/final_df_counts_lbls.pkl"
            )

    def save_as_pkl(
        self,
        out_dict: dict,
        saveas_fp: str,
    ):
        pickle.dump(
            out_dict,
            open(saveas_fp, "wb"),
        )

    def create_data_arrays(
        self,
        toggle_purity_ploidy: str,
        use_gnomad: bool,
    ):
        if use_gnomad:
            predictor_variables = [
                "alt_count",
                "ref_count",
                "read_count",
                "vaf",
                "INFO/gnomad_exomes_AC",
                "INFO/gnomad_exomes_AF",
                "INFO/gnomad_exomes_AC_popmax",
                "INFO/gnomad_exomes_AF_popmax",
                "INFO/gnomad_genomes_AC",
                "INFO/gnomad_genomes_AF",
                "INFO/gnomad_genomes_AC_popmax",
                "INFO/gnomad_genomes_AF_popmax",
            ]
        else:
            predictor_variables = [
                "alt_count",
                "ref_count",
                "read_count",
                "vaf",
            ]

        if toggle_purity_ploidy == "no_pp":
            predictor_variables = predictor_variables
        elif toggle_purity_ploidy == "+purity":
            predictor_variables.extend(["purity"])
        elif toggle_purity_ploidy == "+ploidy":
            predictor_variables.extend(["ploidy"])
        elif toggle_purity_ploidy == "+pp":
            predictor_variables.extend(["purity", "ploidy"])
        else:
            raise ValueError("Choose 1/4: 'no_pp', '+purity', '+ploidy', '+pp'")

        self.apply_init()

        for col in predictor_variables:
            self.merged_df[col] = (
                self.merged_df[col] - self.merged_df[col].mean()
            ) / self.merged_df[col].std()

        g = self.merged_df.groupby("bcr_patient_barcode")

        out = list(
            zip(
                *[
                    [
                        name,
                        values[predictor_variables].values,
                        values["is_somatic"].values[:, np.newaxis],
                        np.unique(values["event"].values)[:, np.newaxis],
                        np.unique(values["OS.time"].values)[:, np.newaxis],
                        np.unique(values["DSS.time"].values)[:, np.newaxis],
                        np.unique(values["DFI.time"].values)[:, np.newaxis],
                        np.unique(values["PFI.time"].values)[:, np.newaxis],
                        np.unique(values["purity"].values)[:, np.newaxis],
                        np.unique(values["ploidy"].values)[:, np.newaxis],
                    ]
                    for name, values in g
                ]
            )
        )

        if toggle_purity_ploidy == "no_pp":
            tcga_id, counts, gs_lbl, event, os, dss, dfi, pfi = out
        elif toggle_purity_ploidy == "+purity":
            tcga_id, counts, gs_lbl, event, os, dss, dfi, pfi, purity = out
        elif toggle_purity_ploidy == "+ploidy":
            tcga_id, counts, gs_lbl, event, os, dss, dfi, pfi, ploidy = out
        elif toggle_purity_ploidy == "+pp":
            tcga_id, counts, gs_lbl, event, os, dss, dfi, pfi, purity, ploidy = out

        pkl_dump_dict = {
            "tcga_id": tcga_id,
            "counts": counts,
            "event": event,
            "OS.time": os,
            "DSS.time": dss,
            "DFI.time": dfi,
            "PFI.time": pfi,
            "is_somatic": gs_lbl,
        }
        if toggle_purity_ploidy == "+purity":
            pkl_dump_dict.update({"purity": purity})
        elif toggle_purity_ploidy == "+ploidy":
            pkl_dump_dict.update({"ploidy": ploidy})
        elif toggle_purity_ploidy == "+pp":
            pkl_dump_dict.update({"purity": purity, "ploidy": ploidy})

        if use_gnomad:
            gnomad = "gad"
        else:
            gnomad = "nogad"

        saveas_fp = f"{self.out_dir}/counts_lbls_{toggle_purity_ploidy}_{gnomad}.pkl"
        self.save_as_pkl(out_dict=pkl_dump_dict, saveas_fp=saveas_fp)

    def create_tmb_counts_arr(self):
        self.apply_init()

        self.merged_df["nonsyn"] = self.merged_df["INFO/BCSQ_csqs"].apply(
            lambda x: 0 if "synonymous" in x else 1
        )

        g = self.merged_df.groupby("bcr_patient_barcode")
        tcga_id, counts, gs_lbl, event, os, dss, dfi, pfi = list(
            zip(
                *[
                    [
                        name,
                        values["nonsyn"].values,
                        values["is_somatic"].values[:, np.newaxis],
                        np.unique(values["event"].values)[:, np.newaxis],
                        np.unique(values["OS.time"].values)[:, np.newaxis],
                        np.unique(values["DSS.time"].values)[:, np.newaxis],
                        np.unique(values["DFI.time"].values)[:, np.newaxis],
                        np.unique(values["PFI.time"].values)[:, np.newaxis],
                    ]
                    for name, values in g
                ]
            )
        )
        pkl_dump_dict = {
            "tcga_id": tcga_id,
            "counts": counts,
            "event": event,
            "OS.time": os,
            "DSS.time": dss,
            "DFI.time": dfi,
            "PFI.time": pfi,
            "is_somatic": gs_lbl,
        }
        saveas_fp = f"{self.out_dir}/tmb_counts_lbls.pkl"
        self.save_as_pkl(out_dict=pkl_dump_dict, saveas_fp=saveas_fp)
