import os
import time
import pandas as pd

from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC as SVMClassifier
from sklearn.linear_model import LogisticRegression as LogisticRegressionClassifier 
from sklearn.ensemble import HistGradientBoostingClassifier

from Aset.log import append_log
from Aset.data_labeling import run_data_labeling
from Aset.data_preparation import prepare_folds
from Aset.ml_algorithms import run_ml_algorithms

# ========================== PARAMETER ==========================
script_dir = os.path.dirname(os.path.abspath(__file__))

dataFolder_path = os.path.join(script_dir, "Data")
resultFolder_path = os.path.join(script_dir, "Results_Baseline_Holdout")

data_file = "Main_Data_new.csv"          # Pastikan file ini sudah ada

target_columns = ["r_dicho", "b_dicho"]
index_columns = ["CompanyId", "PersianYear"]
labeling_columns = ["Return", "Beta"]

random_state = 42
test_size = 0.25
preprocessing_method = "MinMax"        # ganti ke "MinMax" jika mau

# ========================== FOLDER OUTPUT ==========================
dl_output = os.path.join(resultFolder_path, "1_Labeled")
dp_output = os.path.join(resultFolder_path, "2_Prepared")
ml_output = os.path.join(resultFolder_path, "3_Results")

log_path = resultFolder_path
os.makedirs(log_path, exist_ok=True)

# ========================== LOGGING ==========================
append_log(log_path, "baseline_holdout", f"{'='*70}")
append_log(log_path, "baseline_holdout", "BASELINE HOLD-OUT EXPERIMENT")
append_log(log_path, "baseline_holdout", f"Random State     : {random_state}")
append_log(log_path, "baseline_holdout", f"Test Size        : {test_size}")
append_log(log_path, "baseline_holdout", f"Preprocessing    : {preprocessing_method}")
append_log(log_path, "baseline_holdout", f"FP Augmentation  : TIDAK DIGUNAKAN (Baseline)")
append_log(log_path, "baseline_holdout", f"{'='*70}")

start_time = time.time()

# ========================== 1. DATA LABELING ==========================
append_log(log_path, "baseline_holdout", "1. Running Data Labeling...")

run_data_labeling(
    input_path=dataFolder_path,
    output_path=dl_output,
    file_name=data_file.replace('.csv', ''),
    rules=["R2", "B2"],
    data_folder_path=dataFolder_path,
    sort_columns=["CompanyId", "PersianYear"],
    log_path=log_path
)

# ========================== 2. DATA PREPARATION (HOLD-OUT) ==========================
append_log(log_path, "baseline_holdout", "2. Running Data Preparation (Hold-out)...")

prepare_folds(
    input_path=dl_output,
    output_path=dp_output,
    file_name=data_file.replace('.csv', ''),
    target_columns=target_columns,
    index_columns=index_columns,
    labeling_columns=labeling_columns,
    preprocessing_method=preprocessing_method,
    random_state=random_state,
    test_size=test_size,
    method="holdout",                    # Hold-out
    group_column="CompanyId",            # Grouping berdasarkan Company
    log_path=log_path
)

# ========================== 3. MACHINE LEARNING ==========================
append_log(log_path, "baseline_holdout", "3. Running Machine Learning Models...")

# Daftar algoritma untuk baseline
algorithms = {
    "DT": {
        "estimator": DecisionTreeClassifier(),
        "param_grid": {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [10, 20, 50],
            'max_features': [None, 'sqrt', 'log2'],
            'class_weight': ["balanced"]
        }
    },
    "LR": {
        "estimator": LogisticRegressionClassifier(),
        "param_grid": {
            'C': [0.01, 0.1, 1],
            'max_iter': [100, 200, 500, 1000],
            'class_weight': ["balanced"]
        }
    },
    "RF": {
        "estimator": RandomForestClassifier(),
        "param_grid": {
            'n_estimators': [10, 20, 50],
            'criterion': ['gini', 'entropy'],
            'max_depth': [10, 20, 50],
            'max_features': [None, 'sqrt', 'log2'],
            'class_weight': ["balanced"]
        }
    },    
    "SVM":{
        "estimator": SVMClassifier(),
        "param_grid": {
            'C': [0.01, 0.1, 1],
            'kernel': ['linear', 'rbf'],
            'degree': [2, 3],
            'probability': [True],
            'class_weight': ["balanced"]
        }
    },
    "HGB": {
        "estimator": HistGradientBoostingClassifier(),
        "param_grid": {
            'loss': ['log_loss'],
            'learning_rate': [0.01, 0.1, 1],
            'max_depth': [10, 20, 50],
            'class_weight': ["balanced"]
        }
    }
}

run_ml_algorithms(
    input_path=dp_output,
    output_path=ml_output,
    file_name=data_file.replace('.csv', ''),
    target_columns=target_columns,
    algorithms=algorithms,
    scorer='accuracy',
    n_jobs=-1,
    preprocessing_method=preprocessing_method,
    random_state=random_state,
    number_of_splits=1,           # Hold-out hanya 1 split
    kf=None,
    group_column="CompanyId",
    save_models=True,
    log_path=log_path
)

# ========================== FINISH ==========================
end_time = time.time()
elapsed = round(end_time - start_time, 2)

append_log(log_path, "baseline_holdout", f"\nBASELINE HOLD-OUT EXPERIMENT SELESAI!")
append_log(log_path, "baseline_holdout", f"Total Waktu: {elapsed} detik")

print(f"\nBaseline Hold-out selesai! Waktu eksekusi: {elapsed} detik")