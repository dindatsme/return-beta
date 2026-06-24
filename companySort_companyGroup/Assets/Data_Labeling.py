import os
import pandas as pd

class Data_Labeling:
    def __init__(self, input_path, output_path, Log):
        self.iPath = input_path
        self.oPath = output_path
        self.Log = Log

    def __call__(self, file_name, rules, dataFolder_path, sort_columns):
        self.Log(f'\n..:: Start Data_Labeling \"{file_name}\" by \"{rules}\" ::..')
        df = pd.read_csv(f'{self.iPath}/{file_name}.csv')
        Iran_Inflation = pd.read_csv(f'{dataFolder_path}/Iran_Inflation.csv')

        self.create_outPutFolder(self.oPath)
        
        tmp_df = pd.DataFrame()

        for i in df["CompanyId"].unique():
            tmp_company = df[df["CompanyId"]==i].reset_index(drop=True)
            tmp_r_dicho = []
            tmp_b_dicho = []
            for i in range(0, len(tmp_company)):
                if("R2" in rules):
                    tmp_return = tmp_company.loc[i, "Return"]
                    tmp_PersianYear = tmp_company.loc[i, "PersianYear"]
                    tmp_Inflation = Iran_Inflation[Iran_Inflation["PersianYear"]==tmp_PersianYear]["Rate"].iloc[0]
                    if tmp_return > tmp_Inflation:
                        tmp_r_dicho.append(1)
                    else:
                        tmp_r_dicho.append(-1)

                if("B2" in rules):
                    tmp_beta = tmp_company.loc[i, "Beta"]
                    if(tmp_beta > 0 and tmp_beta <= 1):
                        tmp_b_dicho.append(+1)
                    else:
                        tmp_b_dicho.append(-1)

            # tmp_company.drop(tmp_company.tail(1).index, inplace=True)
            if("R2" in rules):
                tmp_company["r_dicho"] = tmp_r_dicho
            if("B2" in rules):
                tmp_company["b_dicho"] = tmp_b_dicho
            
            tmp_df = pd.concat([tmp_df, tmp_company])

        tmp_df = tmp_df.sort_values(by=sort_columns)

        tmp_df.to_csv(f'{self.oPath}/{file_name}.csv', index=False) 
        self.Log(f'..:: End Data_Labeling \"{file_name}\" by \"{rules}\" ::..')

    
    def create_outPutFolder(self, tmp_path):
        if not os.path.isdir(tmp_path):
            os.makedirs(tmp_path)
        
    