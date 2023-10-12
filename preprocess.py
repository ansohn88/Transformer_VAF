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


def sync_dfs(ca: pd.DataFrame, cna: pd.DataFrame, pred_vars: list, saveas: str) -> None:
    # get hugo symbols from both ca & cna files
    hs1 = cna["Hugo_Symbol"].tolist()
    hs2 = ca["Hugo_Symbol"].value_counts().index.tolist()

    # get intersection of hugo symbols from both ca & cna files
    result = Counter(hs1) & Counter(hs2)
    intersected_list = list(result.elements())

    # subset both ca & cna files with the overlapping hugo symbols
    cna = cna.loc[cna["Hugo_Symbol"].isin(intersected_list)]
    ca = ca.loc[ca["Hugo_Symbol"].isin(intersected_list)]

    # get tumor_sample_barcodes from CNA file
    tsb_cna = cna.columns.tolist()[1:]

    # line-up ca tumor_sample_barcodes with cna tumor_sample_barcodes
    ca = ca.loc[ca["Tumor_Sample_Barcode"].isin(tsb_cna)]

    # create patient_id column
    ca["PATIENT_ID"] = ca.apply(
        lambda x: "-".join(x["Tumor_Sample_Barcode"].split("-")[:4])
        if len(x["Tumor_Sample_Barcode"].split("-")) == 6
        else "-".join(x["Tumor_Sample_Barcode"].split("-")[:3]),
        axis=1,
    )
    ca["vaf"] = ca["t_alt_count"] / (ca["t_ref_count"] + ca["t_alt_count"])

    ca_sub = ca[pred_vars]

    ca_sub.to_csv(saveas, sep="\t", index=False)


def fill_cna(ca: pd.DataFrame, cna: pd.DataFrame) -> pd.DataFrame:
    ca.insert(8, "CNA", 0.0)
    for i in range(len(ca)):
        hugo = ca.loc[i, "Hugo_Symbol"]
        tsb = ca.loc[i, "Tumor_Sample_Barcode"]
        # print(i, hugo, tsb)
        cna_val = cna.loc[cna["Hugo_Symbol"] == hugo][tsb].item()
        ca.loc[i, "CNA"] = cna_val
    return ca


def prepare_data(ca_type: str, parent_dir: str) -> None:
    ca_df = read_csv(f"{parent_dir}/{ca_type}/data_mutations_extended.txt")
    cna_df = read_csv(f"{parent_dir}/{ca_type}/data_CNA.txt")

    pred_vars = [
        "PATIENT_ID",
        "Tumor_Sample_Barcode",
        "Hugo_Symbol",
        # "Entrez_Gene_Id",
        "Variant_Classification",
        "t_ref_count",
        "t_alt_count",
        "vaf",
        "Polyphen_Prediction",
        # "Polyphen_Score",
    ]

    saveas = f"{parent_dir}/{ca_type}/{ca_type}.csv"
    sync_dfs(
        ca=ca_df,
        cna=cna_df,
        pred_vars=pred_vars,
        saveas=saveas,
    )

    new_ca_df = read_csv(saveas)
    new_ca_df = fill_cna(ca=new_ca_df, cna=cna_df)
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
