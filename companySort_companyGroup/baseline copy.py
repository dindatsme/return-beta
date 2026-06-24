import os, time
import re
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC as SVMClassifier
from sklearn.linear_model import LogisticRegression as LogisticRegressionClassifier 
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
except ImportError as e:
    raise ImportError(
        "mlxtend is required for frequent pattern augmentation. "
        "Install it with `pip install mlxtend`."
    ) from e

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from Assets.Log import Log
from Assets.Data_Labeling import Data_Labeling
from Assets.Data_Preparation import Data_Preparation
from Assets.ML_Algoritms import ML_Algoritms


def sanitize_feature_name(value):
    return re.sub(r'[^0-9a-zA-Z_]', '_', value)


def build_transaction_list(df, thresholds=None):
    columns = df.columns.tolist()
    if thresholds is None:
        thresholds = {}
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                thresholds[col] = float(df[col].median())
            else:
                thresholds[col] = None

    transactions = []
    for _, row in df.iterrows():
        items = []
        for col in columns:
            val = row[col]
            if thresholds[col] is None:
                items.append(f"{col}={val}")
            else:
                if pd.isna(val):
                    items.append(f"{col}_MISSING")
                elif val > thresholds[col]:
                    items.append(f"{col}_GT_{thresholds[col]:.6g}")
                else:
                    items.append(f"{col}_LE_{thresholds[col]:.6g}")
        transactions.append(items)
    return transactions, thresholds


def encode_transactions(transactions, encoder=None):
    if encoder is None:
        encoder = TransactionEncoder()
        te_array = encoder.fit_transform(transactions)
    else:
        te_array = encoder.transform(transactions)
    return pd.DataFrame(te_array, columns=encoder.columns_), encoder


def mine_frequent_patterns(transaction_df, min_support, max_len, top_k, method):
    if method == "fpgrowth":
        itemsets = fpgrowth(transaction_df, min_support=min_support, use_colnames=True, max_len=max_len)
    else:
        itemsets = apriori(transaction_df, min_support=min_support, use_colnames=True, max_len=max_len)

    if itemsets.empty:
        return itemsets

    itemsets["length"] = itemsets["itemsets"].apply(len)
    # if itemsets["length"].max() >= 2:
    #     itemsets = itemsets[itemsets["length"] >= 2]

    itemsets = itemsets.sort_values(["support", "length"], ascending=[False, False])
    return itemsets.head(top_k)


def mine_association_rules(itemsets_df, min_confidence, top_k):
    if itemsets_df.empty:
        return itemsets_df
    rules = association_rules(itemsets_df, metric="confidence", min_threshold=min_confidence, support_only=True)
    if rules.empty:
        return rules
    return rules.sort_values(["confidence", "lift"], ascending=[False, False]).head(top_k)


def make_augmented_binary_features(df, encoder, thresholds, itemsets, rules):
    transactions, _ = build_transaction_list(df, thresholds=thresholds)
    binary_df, _ = encode_transactions(transactions, encoder=encoder)

    augmented = pd.DataFrame(index=df.index)
    for idx, itemset in enumerate(itemsets, start=1):
        feature_name = f"fp_{idx}_{sanitize_feature_name('_'.join(sorted(itemset)))}"
        augmented[feature_name] = binary_df[list(itemset)].all(axis=1).astype(int)

    for idx, antecedent in enumerate(rules, start=1):
        feature_name = f"rule_{idx}_{sanitize_feature_name('_'.join(sorted(antecedent)))}"
        augmented[feature_name] = binary_df[list(antecedent)].all(axis=1).astype(int)

    return augmented


def augment_frequent_pattern_features(
        df_train, df_test, unwanted_columns,
        min_support, max_itemset_len, top_itemsets,
        use_rules, min_rule_confidence, top_rules,
        mining_method, log_obj=None
    ):
    train_values = df_train.drop(unwanted_columns, axis=1)
    test_values = df_test.drop(unwanted_columns, axis=1)

    train_transactions, thresholds = build_transaction_list(train_values)
    train_matrix, encoder = encode_transactions(train_transactions)

    if log_obj is not None:
        log_obj(f"==> Transaction dataset shape for frequent pattern mining: {train_matrix.shape}")

    itemsets_df = mine_frequent_patterns(
        train_matrix, min_support, max_itemset_len,
        top_itemsets, mining_method
    )

    selected_itemsets = [set(row) for row in itemsets_df["itemsets"].tolist()] if not itemsets_df.empty else []
    selected_rules = []
    if use_rules:
        rules_df = mine_association_rules(itemsets_df, min_rule_confidence, top_rules)
        selected_rules = [set(row) for row in rules_df["antecedents"].tolist()] if not rules_df.empty else []

    if log_obj is not None:
        log_obj(f"==> Frequent itemsets selected: {len(selected_itemsets)}")
        log_obj(f"==> Association rules antecedents selected: {len(selected_rules)}")

    if not selected_itemsets and not selected_rules:
        return df_train, df_test

    train_augmented = make_augmented_binary_features(train_values, encoder, thresholds, selected_itemsets, selected_rules)
    test_augmented = make_augmented_binary_features(test_values, encoder, thresholds, selected_itemsets, selected_rules)

    for col in train_augmented.columns:
        df_train[col] = train_augmented[col].values
        df_test[col] = test_augmented[col].values

    return df_train, df_test

activations = {
    "Data_Labeling": False,
    "Data_Preparation": False,
    "ML_Algoritms": False
}
activations["Data_Labeling"] = True
activations["Data_Preparation"] = True
activations["ML_Algoritms"] = True

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

######### Main Parameters #########
dataFolder_path = os.path.join(script_dir, "Data")
resultFolder_path = os.path.join(script_dir, "Results_ModelB")
dataFiles_name = "Main_Data"
number_of_splits = 5
kfold_method = "SGKF"
sort_columns = ["CompanyId", "PersianYear"]
group_column = "CompanyId"
random_state_numbers = [
    7
]
index_columns = ["CompanyId", "PersianYear"]
target_columns = ["r_dicho", "b_dicho"]
drop_columns = []
######### Main Parameters #########

######### Data_Labeling Parameters #########
dl_input_path = dataFolder_path
dl_output_path = f'{resultFolder_path}/1_Lebeled_Data'
dl_columns = ["Return", "Beta"]
dl_rules = "R2B2"
######### Data_Labeling Parameters #########

######### Data_Preparation Parameters #########
dp_input_path = dl_output_path
dp_output_path = f'{resultFolder_path}/2_Prepared_Data'
dp_methods = [
    "MinMax", "Standard"
]
######### Data_Preparation Parameters #########

######### ML_Algorithms Parameters #########
ml_input_path = dp_output_path
ml_output_path = f'{resultFolder_path}/3_Algorithms_Results'
ml_save_models_flag = False
ml_algorithms = {
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
ml_score_names = [
    'cross_validation_accuracy_mean',
    'test_accuracy_mean','specificity_mean', 
    'sensitivity_mean','mcc_mean',
    'f1_mean', 'auc_roc_mean', 'aupr_mean'
]
ml_nJobs = -1
ml_scorer = 'accuracy'

######### Frequent Pattern Augmentation Parameters #########
fp_mining_method = "fpgrowth"  # or "apriori"
fp_min_support = 0.15
fp_max_itemset_len = 3
fp_top_itemsets = 8
fp_use_association_rules = True
fp_min_rule_confidence = 0.7
fp_top_rules = 8
######### Frequent Pattern Augmentation Parameters #########

######### Algorithms Parameters #########

if not os.path.isdir(resultFolder_path):
    os.makedirs(resultFolder_path)

# get the start time
start_time = time.time()
mainB_log = Log(resultFolder_path, "mainB_log")


######### Log Parameters #########
# Main
mainB_log(F'\n..:: Log Parameters ::..')
mainB_log(F'==> dataFolder_path: {dataFolder_path}')
mainB_log(F'==> resultFolder_path: {resultFolder_path}')
mainB_log(F'==> dataFiles_name: {dataFiles_name}')
mainB_log(F'==> number_of_splits: {number_of_splits}')
mainB_log(F'==> kfold_method: {kfold_method}')
mainB_log(F'==> sort_columns: {sort_columns}')
mainB_log(F'==> group_column: {group_column}')
mainB_log(F'==> random_state_numbers: {random_state_numbers}')
mainB_log(F'==> index_columns: {index_columns}')
mainB_log(F'==> target_columns: {target_columns}')
mainB_log(F'==> drop_columns: {drop_columns}')

# Data_Labeling
mainB_log(F'==> dl_input_path: {dl_input_path}')
mainB_log(F'==> dl_output_path: {dl_output_path}')
mainB_log(F'==> dl_columns: {dl_columns}')
mainB_log(F'==> dl_rules: {dl_rules}')

# Preparing
mainB_log(F'==> dp_input_path: {dp_input_path}')
mainB_log(F'==> dp_output_path: {dp_output_path}')
mainB_log(F'==> dp_methods: {dp_methods}')

# Algorithms
mainB_log(F'==> ml_input_path: {ml_input_path}')
mainB_log(F'==> ml_output_path: {ml_output_path}')
mainB_log(F'==> ml_save_models_flag: {ml_save_models_flag}')
mainB_log(F'==> ml_algorithms: {ml_algorithms}')
mainB_log(F'==> ml_score_names: {ml_score_names}')
mainB_log(F'==> ml_nJobs: {ml_nJobs}')
mainB_log(F'==> ml_scorer: {ml_scorer}')
######### Log Parameters #########
    
if activations["Data_Labeling"]:
    DL = Data_Labeling(dl_input_path, dl_output_path, mainB_log)

if activations["Data_Preparation"]:
    DP = Data_Preparation(dp_input_path, dp_output_path, mainB_log)

if activations["ML_Algoritms"]:
    ML = ML_Algoritms(
        ml_input_path, ml_output_path, ml_save_models_flag,
        ml_algorithms, ml_scorer, ml_score_names, ml_nJobs, mainB_log
    )

######### Main Loop #########

### Beberapa pembersihan data awal
df = pd.read_csv(f'{dataFolder_path}/{dataFiles_name}.csv')
mainB_log(f'\n==> \"{dataFiles_name}\" (Orginal) shape: {df.shape}')
new_df = df.dropna()
new_df = new_df.drop(columns=drop_columns)
new_df = new_df.astype({'Return_ratio_without_risk': 'float64'})
new_df = new_df.sort_values(by=sort_columns)

# Perusahaan dengan titik data lebih dari 3 tahun.
wanted_companies = [
    companyID for companyID in new_df["CompanyId"].unique()
    if len(new_df[new_df["CompanyId"]==companyID]) > 3
] 
new_df = new_df[new_df["CompanyId"].isin(wanted_companies)]

mainB_log(f'==> \"{dataFiles_name}_new\" (After droping drop_columns and Nan rows) shape: {new_df.shape}')
new_df.to_csv(f'{dataFolder_path}/{dataFiles_name}_new.csv', index=False)

file_name = dataFiles_name+'_new'
nRows, nColumns = new_df.shape
number_of_features_original = (nColumns - len(index_columns+dl_columns))
mainB_log(f'==> \"{file_name}_new\" number_of_features (Original): {number_of_features_original}')

# Data_Labeling
if activations["Data_Labeling"]:
    DL(file_name, dl_rules, dataFolder_path, sort_columns)

for random_state in random_state_numbers:
    for dp_method in dp_methods: 
        shuffle_flag = True
        if random_state == None:
            shuffle_flag = False

        if kfold_method == "SKF":
            kf = StratifiedKFold(
                n_splits=number_of_splits,
                shuffle=shuffle_flag,
                random_state=random_state
            )
        elif kfold_method == "SGKF":
            kf = StratifiedGroupKFold(
                n_splits=number_of_splits,
                shuffle=shuffle_flag,
                random_state=random_state
            )

        # 1. Jalankan Data_Preparation Menggunakan Jumlah Fitur Asli
        if activations["Data_Preparation"]:
            DP(
                f'{file_name}', number_of_features_original, target_columns,
                index_columns, dl_columns, dp_method,
                random_state, kf, group_column
            )

            # 2. Jalankan Proses Injeksi Frequent Pattern (Model B) Per Fold
            max_augmented_features = 0 # Untuk mencatat jumlah fitur baru yang terbentuk
            
            for target in target_columns:
                target_path = f'{dp_output_path}/{target}/{random_state}_{dp_method}'
                train_folder = f'{target_path}/train_folds'
                test_folder = f'{target_path}/test_folds'

                for fold_number in range(1, number_of_splits + 1):
                    train_file = f'{train_folder}/fold_{fold_number}.csv'
                    test_file = f'{test_folder}/fold_{fold_number}.csv'

                    df_train = pd.read_csv(train_file)
                    df_test = pd.read_csv(test_file)

                    # Terapkan mining hanya di train, lalu transform ke test
                    df_train, df_test = augment_frequent_pattern_features(
                        df_train, df_test,
                        index_columns + dl_columns + target_columns,
                        fp_min_support, fp_max_itemset_len,
                        fp_top_itemsets, fp_use_association_rules,
                        fp_min_rule_confidence, fp_top_rules,
                        fp_mining_method, mainB_log
                    )

                    # Hitung jumlah fitur baru setelah di-augmentasi (dikurangi kolom non-fitur)
                    current_features = len(df_train.columns) - len(index_columns + dl_columns + target_columns)
                    if current_features > max_augmented_features:
                        max_augmented_features = current_features

                    df_train.to_csv(train_file, index=False)
                    df_test.to_csv(test_file, index=False)
            
            mainB_log(f'==> Total Fitur Setelah Di-Augmentasi (Model B): {max_augmented_features}')

        # 3. Jalankan ML_Algorithms Menggunakan Jumlah Fitur Baru yang Sudah Di-update
        if activations["ML_Algoritms"]:
            # Jika Data_Preparation menyala, gunakan total fitur augmented yang dinamis
            features_to_use = max_augmented_features if activations["Data_Preparation"] else number_of_features_original
            
            ML(
                f'{file_name}', features_to_use,   # <--- SEKARANG SUDAH AMAN (Membaca kolom baru)
                target_columns, dp_method, number_of_splits,
                random_state, kf, group_column
            )

######### Main Loop #########

# get the end time
end_time = time.time()

# get the execution time
elapsed_time = end_time - start_time
mainB_log(f'\n==> Execution time: {elapsed_time} seconds')

mainB_log.write_log()
print("end")