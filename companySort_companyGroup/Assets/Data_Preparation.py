import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression


class Data_Preparation:
    def __init__(self, input_path, output_path, Log):
        self.iPath = input_path
        self.oPath = output_path
        self.Log = Log
        
    def __call__(
            self, file_name, n_features, target_columns,
            index_columns, labeling_columns, preprocessing_method,
            random_state, kf, group_column
    ):
        
        for target in target_columns:
            self.Log(f'\n..:: Start Data_Preparation \"{file_name}\" for {target} with '+
                        f'(random_state: {random_state}, preprocessing_method: {preprocessing_method}) ::..')
            
            df = pd.read_csv(f'{self.iPath}/{file_name}.csv')
            self.Log(f'==> \"{file_name}\" shape: {df.shape}')
            
            tmp_path = f'{self.oPath}/{target}/{random_state}_{preprocessing_method}'
            train_folds_path = f'{tmp_path}/train_folds'
            test_folds_path = f'{tmp_path}/test_folds'

            self.create_outPutFolder(train_folds_path)
            self.create_outPutFolder(test_folds_path)
            
            unWanted_columns = index_columns+labeling_columns+target_columns
            X = df.drop(unWanted_columns, axis=1)
            y = df[target]

            if group_column != None:
                wantedGroups = df[group_column]
                for fold_number, (train_index, test_index) in enumerate(kf.split(X, y, groups=wantedGroups)):

                    # Splitting
                    df_train = df.iloc[train_index]
                    df_test = df.iloc[test_index]
                    self.Log(
                        f'==> fold_{fold_number+1} >> ' +
                        f'\"train\" shape: {df_train.shape}, '+
                        f'\"test\" shape: {df_test.shape}'
                    )

                    train_values = df_train.drop(unWanted_columns, axis=1)
                    test_values = df_test.drop(unWanted_columns, axis=1)

                    # Scaling
                    copy_train_values = train_values.copy()
                    copy_test_values = test_values.copy()

                    if  preprocessing_method == "MinMax":
                        MinMax_scaler = MinMaxScaler().fit(copy_train_values)
                        copy_train_values = MinMax_scaler.transform(copy_train_values)
                        copy_test_values = MinMax_scaler.transform(copy_test_values)

                    if preprocessing_method == "Standard":
                        Standard_scaler = StandardScaler().fit(copy_train_values)
                        copy_train_values = Standard_scaler.transform(copy_train_values)
                        copy_test_values = Standard_scaler.transform(copy_test_values)
                    
                    tmp_df_train = pd.DataFrame(copy_train_values, columns=train_values.columns)
                    df_train.loc[:, tmp_df_train.columns] = tmp_df_train.loc[:,tmp_df_train.columns].values
                    
                    tmp_df_test = pd.DataFrame(copy_test_values, columns=test_values.columns)
                    df_test.loc[:, tmp_df_test.columns] = tmp_df_test.loc[:,tmp_df_test.columns].values

                    df_train.to_csv(f'{train_folds_path}/fold_{fold_number+1}.csv', index=False)
                    df_test.to_csv(f'{test_folds_path}/fold_{fold_number+1}.csv', index=False)  
                        
                 
            else:
                for fold_number, (train_index, test_index) in enumerate(kf.split(X, y)):

                    # Splitting
                    df_train = df.iloc[train_index]
                    df_test = df.iloc[test_index]
                    self.Log(
                        f'==> fold_{fold_number+1} >> ' +
                        f'\"train\" shape: {df_train.shape}, '+
                        f'\"test\" shape: {df_test.shape}'
                    )

                    train_values = df_train.drop(unWanted_columns, axis=1)
                    test_values = df_test.drop(unWanted_columns, axis=1)

                    # Scaling
                    copy_train_values = train_values.copy()
                    copy_test_values = test_values.copy()

                    if  preprocessing_method == "MinMax":
                        MinMax_scaler = MinMaxScaler().fit(copy_train_values)
                        copy_train_values = MinMax_scaler.transform(copy_train_values)
                        copy_test_values = MinMax_scaler.transform(copy_test_values)

                    if preprocessing_method == "Standard":
                        Standard_scaler = StandardScaler().fit(copy_train_values)
                        copy_train_values = Standard_scaler.transform(copy_train_values)
                        copy_test_values = Standard_scaler.transform(copy_test_values)
                    
                    tmp_df_train = pd.DataFrame(copy_train_values, columns=train_values.columns)
                    df_train.loc[:, tmp_df_train.columns] = tmp_df_train.loc[:,tmp_df_train.columns].values
                    
                    tmp_df_test = pd.DataFrame(copy_test_values, columns=test_values.columns)
                    df_test.loc[:, tmp_df_test.columns] = tmp_df_test.loc[:,tmp_df_test.columns].values

                    df_train.to_csv(f'{train_folds_path}/fold_{fold_number+1}.csv', index=False)
                    df_test.to_csv(f'{test_folds_path}/fold_{fold_number+1}.csv', index=False)  
                    
            self.Log(f'..:: End Data_Preparation \"{file_name}\" for {target} with '+
                    f'(random_state: {random_state}, Scaling_method: {preprocessing_method}) ::..')
            
      
    def get_finalScores(self, features_lists, features):
        final_scores = pd.DataFrame(data = {'Features':features})

        for i in range(len(features_lists)):
            tmp_scores = []
            for feature in final_scores["Features"]:
               tmp_scores.append(features_lists[i].index(feature)+1)
            final_scores[i+1] = tmp_scores

        modes = []
        means = []
        stds = []
        for i in range(len(final_scores)):
            data = final_scores.iloc[i, 1:]
            modes.append(data.mode()[0])
            means.append(data.mean())
            stds.append(data.std())
        final_scores['Mode'] = modes
        final_scores['Mean'] = means
        final_scores['Std'] = stds
        
        final_scores = final_scores.sort_values(by=['Mode', 'Mean', "Std"])
        final_scores.reset_index(drop=True, inplace=True)
        return final_scores
    

    def create_outPutFolder(self, tmp_path):
        if not os.path.isdir(tmp_path):
            os.makedirs(tmp_path)