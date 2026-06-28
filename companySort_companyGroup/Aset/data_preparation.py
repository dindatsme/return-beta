import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, GroupKFold, GroupShuffleSplit, StratifiedGroupKFold
from typing import Optional

from Aset.log import append_log


def create_output_folder(path: str):
    os.makedirs(path, exist_ok=True)

def prepare_folds(
    input_path: str,
    output_path: str,
    file_name: str,
    target_columns: list,
    index_columns: list,
    labeling_columns: list,
    preprocessing_method: str = "Standard",
    random_state: int = 42,
    n_splits: int = 5,
    test_size: float = 0.25,
    method: str = "cv",                     # "cv" atau "holdout"
    group_column = None,
    kf=None,                                # ← Tambahkan ini
    log_path: Optional[str] = None,
    log_filename: str = "app_log"
):
    """
    Fungsi Data Preparation - Support CV dan Holdout
    """
    if log_path is None:
        log_path = output_path

    for target in target_columns:
        append_log(
            log_path, log_filename,
            f'..:: Start Data_Preparation "{file_name}" for {target} '
            f'({method.upper()} | random_state: {random_state} | '
            f'preprocessing: {preprocessing_method}) ::..'
        )

        df = pd.read_csv(os.path.join(input_path, f"{file_name}.csv"))
        append_log(log_path, log_filename, f'==> "{file_name}" shape: {df.shape}')

        base_path = os.path.join(output_path, target, f"{random_state}_{preprocessing_method}")
        train_path = os.path.join(base_path, "train")
        test_path = os.path.join(base_path, "test")

        create_output_folder(train_path)
        create_output_folder(test_path)

        unwanted = index_columns + labeling_columns + target_columns
        X = df.drop(unwanted, axis=1)
        y = df[target]

        if isinstance(group_column, list):
            group_column = group_column[0] if len(group_column) == 1 else None
        groups = df[group_column].squeeze() if group_column and group_column in df.columns else None

        # ==================== SPLITTING LOGIC ====================
        if method.lower() == "holdout":
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=random_state
            )
            # groups = df[group_column].squeeze() if group_column else None
            split_iterator = splitter.split(X, y, groups=groups)
            append_log(log_path, log_filename, f"Using Hold-out (test_size={test_size})")

        else:  # CV mode
            if kf is not None:
                # Pakai kf yang dikirim dari luar (recommended)
                splitter = kf
                # groups = df[group_column].squeeze() if group_column and 'Group' in kf.__class__.__name__ else None
                if groups is not None:
                    split_iterator = splitter.split(X, y, groups=groups)
                else:
                    split_iterator = splitter.split(X, y)
                append_log(log_path, log_filename, f"Using {kf.__class__.__name__} (n_splits={n_splits})")
            else:
                # Fallback
                if group_column and group_column in df.columns:
                    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                    split_iterator = splitter.split(X, y, groups=groups)
                    append_log(log_path, log_filename, f"Using StratifiedGroupKFold (n_splits={n_splits})")
                else:
                    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                    split_iterator = splitter.split(X, y)
                    append_log(log_path, log_filename, f"Using KFold (n_splits={n_splits})")

        # ==================== PROCESS EACH SPLIT ====================
        for fold_number, (train_index, test_index) in enumerate(split_iterator):
            df_train = df.iloc[train_index].copy()
            df_test = df.iloc[test_index].copy()

            append_log(
                log_path, log_filename,
                f'==> {"Holdout" if method.lower()=="holdout" else f"fold_{fold_number+1}"} >> '
                f'train: {df_train.shape}, test: {df_test.shape}'
            )

            train_values = df_train.drop(unwanted, axis=1)
            test_values = df_test.drop(unwanted, axis=1)

            # Scaling
            if preprocessing_method == "MinMax":
                scaler = MinMaxScaler()
            elif preprocessing_method == "Standard":
                scaler = StandardScaler()
            else:
                raise ValueError(f"Preprocessing method '{preprocessing_method}' not supported.")

            train_scaled = scaler.fit_transform(train_values)
            test_scaled = scaler.transform(test_values)

            train_scaled_df = pd.DataFrame(train_scaled, columns=train_values.columns, index=df_train.index)
            test_scaled_df = pd.DataFrame(test_scaled, columns=test_values.columns, index=df_test.index)

            df_train.loc[:, train_scaled_df.columns] = train_scaled_df
            df_test.loc[:, test_scaled_df.columns] = test_scaled_df

            # Save
            if method.lower() == "holdout":
                df_train.to_csv(os.path.join(train_path, "fold_1.csv"), index=False)
                df_test.to_csv(os.path.join(test_path, "fold_1.csv"), index=False)
            else:
                df_train.to_csv(os.path.join(train_path, f"fold_{fold_number+1}.csv"), index=False)
                df_test.to_csv(os.path.join(test_path, f"fold_{fold_number+1}.csv"), index=False)

        append_log(log_path, log_filename, f'..:: End Data_Preparation for {target} ::..')

    print(f"Data Preparation selesai ({method.upper()}) untuk file: {file_name}")
