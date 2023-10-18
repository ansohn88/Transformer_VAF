import pickle
from collections import Counter

import numpy as np
import pandas as pd


def read_csv(fp: str, sep: str = "\t", low_mem: bool = False) -> pd.DataFrame:
    return pd.read_csv(fp, sep=sep, low_memory=low_mem)


def subset_missense_only(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["Variant_Classification"] == "Missense_Mutation"]


def subset_os_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[["PATIENT_ID", "OS_STATUS", "OS_MONTHS"]]


def subset_pfi_m_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[["PATIENT_ID", "PFS_M_ADV_STATUS", "PFS_M_ADV_MONTHS"]]


def reconfigure_cna_df(cna_df: pd.DataFrame) -> pd.DataFrame:
    hugo = cna_df["Hugo_Symbol"].value_counts().index.tolist()
    cna_df = cna_df.drop(["Hugo_Symbol"], axis=1)
    cna_df = cna_df.transpose().reset_index()

    hugo.insert(0, "Tumor_Sample_Barcode")
    cna_df = cna_df.set_axis(hugo, axis=1)
    return cna_df


def fill_cna(ca: pd.DataFrame, cna: pd.DataFrame) -> pd.DataFrame:
    start = time.time()
    ca.insert(8, "CNA", 0.0)

    tsb = cna["Tumor_Sample_Barcode"].value_counts().index.tolist()
    hs = cna.columns.tolist()
    hs.pop(0)

    for i in tsb:
        for j in hs:
            cna_val = cna.loc[cna["Tumor_Sample_Barcode"] == i][j]
            row = ca.loc[(ca["Tumor_Sample_Barcode"] == i) & (ca["Hugo_Symbol"] == j)]
            if len(row) > 0:
                row["CNA"] = cna_val
    print(f"Time elapse to fill CNA: {time.time() - start:.2f} s")
    return ca


def prepare_data(ca_type: str, parent_dir: str) -> None:
    ca_df = read_csv(f"{parent_dir}/{ca_type}/data_mutations_extended.txt")
    # create vaf column
    ca_df["vaf"] = ca_df["t_alt_count"] / (ca_df["t_ref_count"] + ca_df["t_alt_count"])
    
    cna_df = read_csv(f"{parent_dir}/{ca_type}/data_CNA.txt")
    cna_df = reconfigure_cna_df(cna_df=cna_df)

    pred_vars = [
        "PATIENT_ID",
        "Tumor_Sample_Barcode",
        "Hugo_Symbol",
        "Entrez_Gene_Id",
        "Variant_Classification",
        "t_ref_count",
        "t_alt_count",
        "t_depth",
        "vaf",
        "Polyphen_Prediction",
        "Polyphen_Score",
    ]

    ca_df_sub = ca_df[pred_vars]

    new_ca_df = fill_cna(ca=ca_df_sub, cna=cna_df)
    new_ca_df.to_csv(
        f"{parent_dir}/{ca_type}/final_{ca_type}.csv", sep="\t", index=False
    )


def prepare_data_with_label(
    ca_type: str,
    pdir: str = "/home/andy/baraslab/projects/genie_bcp/data",
    s_metric: str = "OS",
) -> None:
    c_df = read_csv(f"{pdir}/{ca_type}/final_{ca_type}.csv")
    l_df = pd.read_csv(
        f"{pdir}/{ca_type}/data_clinical_supp_survival.txt", sep="\t", skiprows=5
    )

    if s_metric == "OS":
        l_df = subset_os_only(l_df)
    elif s_metric == "PFI":
        l_df = subset_pfi_m_only(l_df)
    l_df = l_df.drop_duplicates(subset="PATIENT_ID", keep="last")

    merged = pd.merge(c_df, l_df, how="left", on=["PATIENT_ID"])
    merged.to_csv(f"{pdir}/{ca_type}/counts_surv.csv", sep="\t", index=False)


def save_prepared_data(
    pdir: str = "/home/andy/baraslab/projects/genie_bcp/data",
) -> None:
    # CRC
    prepare_data(ca_type="crc", parent_dir=pdir)

    # NSCLC
    prepare_data(ca_type="nsclc", parent_dir=pdir)


def save_prepared_data_with_label(
    pdir: str = "/home/andy/baraslab/projects/genie_bcp/data", s_metric: str = "OS"
) -> None:
    # CRC
    prepare_data_with_label(ca_type="crc", pdir=pdir, s_metric=s_metric)

    # NSCLC
    prepare_data_with_label(ca_type="nsclc", pdir=pdir, s_metric=s_metric)


def create_data_arr(
    df: pd.DataFrame, pred_vars: list, groupby: str, saveas: str, s_metric: str = "OS"
) -> None:
    if s_metric == "OS":
        e = "OS_STATUS"
        t = "OS_MONTHS"
    else:
        e = "PFS_M_ADV_STATUS"
        t = "PFS_M_ADV_MONTHS"

    # fill CNA N/As with 0.0 -- no copy number changes
    df["CNA"] = df["CNA"].fillna(0.0)

    # fill polyphen score N/As with 0.5 (?) -- possibly damaging
    # df["Polyphen_Score"] = df["Polyphen_Score"].fillna(0.5)

    # fill polyphen_prediction N/As with `unknown`
    df["Polyphen_Prediction"] = df["Polyphen_Prediction"].fillna("unknown")

    # convert 1:DECEASED to 1 and 0:LIVING to 0
    df[e] = df[e].apply(lambda x: int(x.split(":")[0]))

    g = df.groupby(groupby)

    out = list(
        zip(
            *[
                [
                    name,
                    values[pred_vars].values,
                    # values["Hugo_Symbol"].values,
                    # values["Variant_Classification"].values,
                    # values["Polyphen_Prediction"].values,
                    np.unique(values[e].values)[:, np.newaxis],
                    np.unique(values[t].values)[:, np.newaxis],
                ]
                for name, values in g
            ]
        )
    )

    sample_id, counts, event, time = out

    pickle.dump(
        {
            "id": sample_id,
            "counts": counts,
            "event": event,
            "time": time,
        },
        open(saveas, "wb"),
    )


def create_combined_data_arr(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    pred_vars: list,
    groupby: str,
    saveas: str,
    keep_csv: bool,
    s_metric: str = "OS",
) -> None:
    if s_metric == "OS":
        e = "OS_STATUS"
        t = "OS_MONTHS"
    else:
        e = "PFS_M_ADV_STATUS"
        t = "PFS_M_ADV_MONTHS"

    # concat dfs vertically
    df = pd.concat([df1, df2], ignore_index=True)

    # fill CNA N/As with 0.0 -- no copy number changes
    df["CNA"] = df["CNA"].fillna(0.0)

    # fill polyphen_prediction N/As with `unknown`
    df["Polyphen_Prediction"] = df["Polyphen_Prediction"].fillna("unknown")

    # convert 1:DECEASED to 1 and 0:LIVING to 0
    df[e] = df[e].apply(lambda x: int(x.split(":")[0]))

    if keep_csv:
        dfc = df.copy(deep=True)
        dfc = dfc[pred_vars]
        df.to_csv(f"{saveas.split('.')[0]}.csv", sep="\t", index=False)

    g = df.groupby(groupby)

    out = list(
        zip(
            *[
                [
                    name,
                    values[pred_vars].values,
                    # values["Hugo_Symbol"].values,
                    # values["Variant_Classification"].values,
                    # values["Polyphen_Prediction"].values,
                    np.unique(values[e].values)[:, np.newaxis],
                    np.unique(values[t].values)[:, np.newaxis],
                ]
                for name, values in g
            ]
        )
    )

    sample_id, counts, event, time = out

    pickle.dump(
        {
            "id": sample_id,
            "counts": counts,
            "event": event,
            "time": time,
        },
        open(saveas, "wb"),
    )


if __name__ == "__main__":
    # save_prepared_data()

    # save_prepared_data_with_label(s_metric="OS")

    # pred_vars = ["t_ref_count", "t_alt_count", "vaf"]
    # pred_vars = ["t_ref_count", "t_alt_count", "vaf", "Hugo_Symbol"]
    # pred_vars = [
    #     "t_ref_count",
    #     "t_alt_count",
    #     "vaf",
    #     "Hugo_Symbol",
    #     "Variant_Classification",
    # ]
    pred_vars = [
        "t_ref_count",
        "t_alt_count",
        "vaf",
        "Hugo_Symbol",
        "Variant_Classification",
        "Polyphen_Prediction",
    ]
    # pred_vars = ["t_ref_count", "t_alt_count", "vaf", "CNA"]
    # pred_vars = ["t_ref_count", "t_alt_count", "vaf", "CNA", "Hugo_Symbol"]
    # pred_vars = [
    #     "t_ref_count",
    #     "t_alt_count",
    #     "vaf",
    #     "CNA",
    #     "Hugo_Symbol",
    #     "Polyphen_Prediction",
    # ]

    # crc = read_csv("/home/andy/baraslab/projects/genie_bcp/data/crc/counts_surv.csv")
    # create_data_arr(
    #     df=crc,
    #     pred_vars=pred_vars,
    #     groupby="Tumor_Sample_Barcode",
    #     saveas="/home/andy/baraslab/projects/genie_bcp/data/crc/final_vaf.pkl",
    # )

    # nsclc = read_csv(
    #     "/home/andy/baraslab/projects/genie_bcp/data/nsclc/counts_surv.csv"
    # )
    # create_data_arr(
    #     df=nsclc,
    #     pred_vars=pred_vars,
    #     groupby="Tumor_Sample_Barcode",
    #     saveas="/home/andy/baraslab/projects/genie_bcp/data/nsclc/final_vaf.pkl",
    # )

    crc = read_csv("/home/andy/baraslab/projects/genie_bcp/data/crc/counts_surv.csv")
    nsclc = read_csv(
        "/home/andy/baraslab/projects/genie_bcp/data/nsclc/counts_surv.csv"
    )
    create_combined_data_arr(
        df1=nsclc,
        df2=crc,
        pred_vars=pred_vars,
        groupby="Tumor_Sample_Barcode",
        keep_csv=True,
        saveas="/home/andy/baraslab/projects/genie_bcp/data/combined/final_vaf_hugo_varc_pp.pkl",
    )
