import os, time
import pandas as pd

from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC as SVMClassifier
from sklearn.linear_model import LogisticRegression as LogisticRegressionClassifier 
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from Assets.Log import Log
from Assets.Data_Labeling import Data_Labeling
from Assets.Data_Preparation import Data_Preparation
from Assets.ML_Algoritms import ML_Algoritms

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
resultFolder_path = os.path.join(script_dir, "Results_ModelA")
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
######### Algorithms Parameters #########

if not os.path.isdir(resultFolder_path):
    os.makedirs(resultFolder_path)

# get the start time
start_time = time.time()
main_log = Log(resultFolder_path, "main_log")


######### Log Parameters #########
# Main
main_log(F'\n..:: Log Parameters ::..')
main_log(F'==> dataFolder_path: {dataFolder_path}')
main_log(F'==> resultFolder_path: {resultFolder_path}')
main_log(F'==> dataFiles_name: {dataFiles_name}')
main_log(F'==> number_of_splits: {number_of_splits}')
main_log(F'==> kfold_method: {kfold_method}')
main_log(F'==> sort_columns: {sort_columns}')
main_log(F'==> group_column: {group_column}')
main_log(F'==> random_state_numbers: {random_state_numbers}')
main_log(F'==> index_columns: {index_columns}')
main_log(F'==> target_columns: {target_columns}')
main_log(F'==> drop_columns: {drop_columns}')

# Data_Labeling
main_log(F'==> dl_input_path: {dl_input_path}')
main_log(F'==> dl_output_path: {dl_output_path}')
main_log(F'==> dl_columns: {dl_columns}')
main_log(F'==> dl_rules: {dl_rules}')

# Preparing
main_log(F'==> dp_input_path: {dp_input_path}')
main_log(F'==> dp_output_path: {dp_output_path}')
main_log(F'==> dp_methods: {dp_methods}')

# Algorithms
main_log(F'==> ml_input_path: {ml_input_path}')
main_log(F'==> ml_output_path: {ml_output_path}')
main_log(F'==> ml_save_models_flag: {ml_save_models_flag}')
main_log(F'==> ml_algorithms: {ml_algorithms}')
main_log(F'==> ml_score_names: {ml_score_names}')
main_log(F'==> ml_nJobs: {ml_nJobs}')
main_log(F'==> ml_scorer: {ml_scorer}')
######### Log Parameters #########
    
if activations["Data_Labeling"]:
    DL = Data_Labeling(dl_input_path, dl_output_path, main_log)

if activations["Data_Preparation"]:
    DP = Data_Preparation(dp_input_path, dp_output_path, main_log)

if activations["ML_Algoritms"]:
    ML = ML_Algoritms(
        ml_input_path, ml_output_path, ml_save_models_flag,
        ml_algorithms, ml_scorer, ml_score_names, ml_nJobs, main_log
    )

######### Main Loop #########

### Some data cleaning
df = pd.read_csv(f'{dataFolder_path}/{dataFiles_name}.csv')
main_log(f'\n==> \"{dataFiles_name}\" (Orginal) shape: {df.shape}')
new_df = df.dropna()
new_df = new_df.drop(columns=drop_columns)
new_df = new_df.astype({'Return_ratio_without_risk': 'float64'})
new_df = new_df.sort_values(by=sort_columns)

# Companies with more than 3 years data points.
wanted_companies = [
    companyID for companyID in new_df["CompanyId"].unique()
    if len(new_df[new_df["CompanyId"]==companyID]) > 3
] 
new_df = new_df[new_df["CompanyId"].isin(wanted_companies)]

main_log(f'==> \"{dataFiles_name}_new\" (After droping drop_columns and Nan rows) shape: {new_df.shape}')
new_df.to_csv(f'{dataFolder_path}/{dataFiles_name}_new.csv', index=False)

file_name = dataFiles_name+'_new'
nRows, nColumns = new_df.shape
number_of_features = (nColumns - len(index_columns+dl_columns))
main_log(f'==> \"{file_name}_new\" number_of_features: {number_of_features}')

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

        # Data_Preparation
        if activations["Data_Preparation"]:
            DP(
                f'{file_name}', number_of_features, target_columns,
                index_columns, dl_columns, dp_method,
                random_state, kf, group_column
            )  

        # ML_Algoritms
        if activations["ML_Algoritms"]:
            ML(
                f'{file_name}', number_of_features,
                target_columns, dp_method, number_of_splits,
                random_state, kf, group_column
          )

######### Main Loop #########

# get the end time
end_time = time.time()

# get the execution time
elapsed_time = end_time - start_time
main_log(f'\n==> Execution time: {elapsed_time} seconds')

main_log.write_log()
print("end")