# data_labeling.py
import os
import pandas as pd
from Aset.log import append_log


def create_output_folder(output_path: str):
    """Buat folder output jika belum ada"""
    os.makedirs(output_path, exist_ok=True)


def load_data(input_path: str, file_name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(input_path, f"{file_name}.csv"))


def load_inflation_data(data_folder_path: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_folder_path, "Iran_Inflation.csv"))


def label_company_data(company_df: pd.DataFrame, 
                      rules: list | str, 
                      inflation_df: pd.DataFrame) -> pd.DataFrame:
    
    tmp_company = company_df.copy().reset_index(drop=True)
    
    tmp_r_dicho = []
    tmp_b_dicho = []

    for i in range(len(tmp_company)):
        row = tmp_company.iloc[i]

        if "R2" in rules:
            inflation_rate = inflation_df[
                inflation_df["PersianYear"] == row["PersianYear"]
            ]["Rate"].iloc[0]
            tmp_r_dicho.append(1 if row["Return"] > inflation_rate else -1)

        if "B2" in rules:
            beta = row["Beta"]
            tmp_b_dicho.append(1 if 0 < beta <= 1 else -1)

    if "R2" in rules:
        tmp_company["r_dicho"] = tmp_r_dicho
    if "B2" in rules:
        tmp_company["b_dicho"] = tmp_b_dicho

    return tmp_company


def run_data_labeling(
    input_path: str,
    output_path: str,
    file_name: str,
    rules: list | str,
    data_folder_path: str,
    sort_columns: list,
    log_path: str,    
    log_filename: str = "app_log"
):
    """
    Fungsi utama Data Labeling menggunakan logger yang sudah dibuat sebelumnya
    """
    if log_path is None:
        log_path = output_path

    append_log(log_path, log_filename, 
               f'..:: Start Data_Labeling "{file_name}" by "{rules}" ::..')

    # Load data
    df = load_data(input_path, file_name)
    iran_inflation = load_inflation_data(data_folder_path)

    create_output_folder(output_path)

    labeled_dfs = []

    for company_id in df["CompanyId"].unique():
        company_df = df[df["CompanyId"] == company_id].reset_index(drop=True)
        labeled_company = label_company_data(company_df, rules, iran_inflation)
        labeled_dfs.append(labeled_company)

    # Merge & sort
    result_df = pd.concat(labeled_dfs, ignore_index=True)
    result_df = result_df.sort_values(by=sort_columns)

    # Simpan hasil
    result_df.to_csv(os.path.join(output_path, f"{file_name}.csv"), index=False)

    append_log(log_path, log_filename, 
               f'..:: End Data_Labeling "{file_name}" by "{rules}" ::..')

    print(f"✅ Data labeling selesai untuk: {file_name}.csv")