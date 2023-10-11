import collections
import pickle

import pandas as pd


def read_csv(fp: str, sep: str = "\t", low_mem: bool = False) -> pd.DataFrame:
    return pd.read_csv(fp, sep=sep, low_memory=low_mem)


def subset_missense_only(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["Variant_Classification"] == "Missense_Mutation"]


def sync_dfs(ca: pd.DataFrame, cna: pd.DataFrame, pred_vars: list, saveas: str) -> None:
    # get tumor_sample_barcodes from CNA file
    tsb_cna = cna.columns.tolist()[1:]

    # get hugo symbols from both ca & cna files
    hs1 = cna["Hugo_Symbol"].tolist()
    hs2 = ca["Hugo_Symbol"].value_counts().index.tolist()

    # get intersection of hugo symbols from both ca & cna files
    result = collections.Counter(hs1) & collections.Counter(hs2)
    intersected_list = list(result.elements())

    # subset both ca & cna files with the overlapping hugo symbols
    cna = cna.loc[cna["Hugo_Symbol"].isin(intersected_list)]
    ca = ca.loc[ca["Hugo_Symbol"].isin(intersected_list)]

    # line-up ca tumor_sample_barcodes with cna tumor_sample_barcodes
    ca = ca.loc[ca["Tumor_Sample_Barcode"].isin(tsb_cna)]

    #
    ca["PATIENT_ID"] = ca.apply(
        lambda x: "-".join(x["Tumor_Sample_Barcode"].split("-")[:-1]), axis=1
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


def prepare_data(ca_type, parent_dir: str) -> None:
    ca_df = read_csv(f"{parent_dir}/{ca_type}/data_mutations_extended.txt")
    cna_df = read_csv(f"{parent_dir}/{ca_type}/data_CNA.txt")

    pred_vars = [
        "PATIENT_ID",
        "Tumor_Sample_Barcode",
        "Hugo_Symbol",
        "Entrez_Gene_Id",
        "Variant_Classification",
        "t_ref_count",
        "t_alt_count",
        "vaf",
        "Polyphen_Score",
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


def create_data_arr(
    df: pd.DataFrame, pred_vars: list, groupby: str, saveas: str
) -> None:
    g = df.groupby(groupby)

    out = list(zip(*[[name, values[pred_vars].values] for name, values in g]))

    sample_id, counts = out

    pickle.dump({"id": sample_id, "counts": counts}, open(saveas, "wb"))


if __name__ == "__main__":
    TODO
