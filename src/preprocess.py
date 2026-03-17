import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from os import PathLike


def substitute_path_root(path, root: str | PathLike):
    p = path.split("/")
    p[0] = root
    p = os.path.join(*p)
    return p


def preprocess_chexpert_dataframe(
        data_root: str | PathLike,
        csv_filename: str | PathLike,
        target_cols: list[str]
) -> pd.DataFrame:
    
    df = pd.read_csv(os.path.join(data_root, csv_filename))

    df = df[df["Frontal/Lateral"] == "Frontal"]
    df["Path"] = df["Path"].apply(substitute_path_root, root=data_root)
    df = df[["Path", "Sex", "Age", "No Finding"] + target_cols]

    for idx, row in df.iterrows():
        if row["No Finding"] == 1:
            df.loc[idx, target_cols] = 0

    df["No Finding"] = df["No Finding"].fillna(0)
    df = df[df[target_cols].notna().any(axis=1)]

    return df