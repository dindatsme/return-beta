import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score,
    auc, matthews_corrcoef
)
from sklearn.model_selection import GridSearchCV
from joblib import dump


class ML_Algoritms:
    def __init__(
            self, input_path, output_path, save_models_flag,
            algorithms, scorer, score_names, n_jobs, Log
    ):
        self.iPath      = input_path
        self.oPath      = output_path
        self.saveModels = save_models_flag
        self.algorithms = algorithms
        self.scorer     = scorer
        self.scNames    = score_names
        self.n_jobs     = n_jobs
        self.alNames    = list(self.algorithms.keys())
        self.Log        = Log

    def __call__(
            self, file_name, target_columns,
            preprocessing_method, number_of_splits,
            random_state, kf, group_column
    ):
        # number_of_features parameter removed — all features in the fold CSV
        # are used (original features + fp_* / rule_* augmented columns).

        for target_column in target_columns:
            self.results = {alName: {} for alName in self.alNames}
            self.models  = {alName: {} for alName in self.alNames}

            self.max_auc_roc = {alName: 0 for alName in self.alNames}
            self.max_aupr    = {alName: 0 for alName in self.alNames}

            result_path = (
                f'{self.oPath}/{target_column}/'
                f'{random_state}_{preprocessing_method}'
            )

            self.create_outPutFolder(result_path)

            self.Log(
                f'\n..:: Start Running ML_Algoritms on "{file_name}" '
                f'for {target_column} '
                f'(random_state: {random_state}, '
                f'preprocessing_method: {preprocessing_method})'
            )

            for alName in self.alNames:
                self.algorithms[alName]["param_grid"]['random_state'] = [random_state]

                all_X_train                  = []
                all_y_train                  = []
                all_X_test                   = []
                all_y_test                   = []
                all_y_pred                   = []
                all_y_proba                  = []
                all_cross_validation_accuracy = []
                all_bestModels               = []
                all_bestParams               = []

                self.Log(f'\n..:: Start Running {alName} Classifier ::..')

                for fold_number in range(1, number_of_splits + 1):

                    tmp_iPath = (
                        f'{self.iPath}/{target_column}/'
                        f'{random_state}_{preprocessing_method}'
                    )
                    df_train = pd.read_csv(
                        f'{tmp_iPath}/train_folds/fold_{fold_number}.csv'
                    )
                    df_test = pd.read_csv(
                        f'{tmp_iPath}/test_folds/fold_{fold_number}.csv'
                    )

                    # Use ALL columns in the CSV as features except the target.
                    # This naturally includes:
                    #   - original scaled features (from Data_Preparation)
                    #   - fp_*   columns (frequent-itemset features)
                    #   - rule_* columns (association-rule antecedent features)
                    # No ranked-feature CSV is needed anymore.
                    selected_features = [
                        col for col in df_train.columns
                        if col != target_column
                    ]

                    X_train = df_train[selected_features]
                    X_test  = df_test[selected_features]
                    y_train = df_train[target_column]
                    y_test  = df_test[target_column]

                    if alName != "HGB":
                        class_weights = compute_class_weight(
                            class_weight="balanced",
                            classes=np.unique(y_train),
                            y=y_train
                        )
                        class_weight_dict = {
                            i: w for i, w in zip(np.unique(y_train), class_weights)
                        }
                        self.algorithms[alName]["param_grid"]['class_weight'] = [
                            class_weight_dict
                        ]
                    else:
                        class_weight_dict = "balanced"

                    all_X_train.append(X_train)
                    all_y_train.append(y_train)
                    all_X_test.append(X_test)
                    all_y_test.append(y_test)

                    # HyperParameter Tuning
                    grid_obj = GridSearchCV(
                        estimator=self.algorithms[alName]["estimator"],
                        param_grid=self.algorithms[alName]["param_grid"],
                        scoring=self.scorer, cv=kf, n_jobs=self.n_jobs,
                        error_score='raise'
                    )

                    sample_weights = compute_sample_weight(
                        class_weight=class_weight_dict, y=y_train
                    )

                    if group_column is not None:
                        wanted_groups = df_train[group_column]
                        grid_obj.fit(
                            X_train, y_train,
                            sample_weight=sample_weights,
                            groups=wanted_groups
                        )
                    else:
                        grid_obj.fit(X_train, y_train, sample_weight=sample_weights)

                    best_est = grid_obj.best_estimator_
                    all_bestModels.append(best_est)

                    y_pred  = best_est.predict(X_test)
                    y_proba = best_est.predict_proba(X_test)
                    all_y_pred.append(y_pred)
                    all_y_proba.append(y_proba)

                    if self.saveModels:
                        dump(
                            best_est,
                            f'{result_path}/by_Algorithms/{alName}/fold{fold_number}.joblib'
                        )

                    self.Log(
                        f'..:: HyperParameter Tuned for Fold {fold_number} ::..\n'
                        f'==> best_params_: {grid_obj.best_params_}\n'
                        f'==> Mean cross-validated "{self.scorer}" of the best_estimator: '
                        f'{self.round_percentage(grid_obj.best_score_)}%'
                    )

                    all_bestParams.append(grid_obj.best_params_)
                    all_cross_validation_accuracy.append(grid_obj.best_score_)

                self.calc_results(
                    alName, all_y_test, all_y_pred, all_y_proba,
                    all_cross_validation_accuracy, all_bestParams
                )
                self.Log(self.get_algorithmLog(alName))

                if (
                    self.results[alName]['aupr_mean']    > self.max_aupr[alName] or
                    self.results[alName]['auc_roc_mean'] > self.max_auc_roc[alName]
                ):
                    self.plot_algorithmResults(
                        alName, all_y_test, all_y_proba, result_path
                    )
                    self.max_aupr[alName]    = self.results[alName]['aupr_mean']
                    self.max_auc_roc[alName] = self.results[alName]['auc_roc_mean']

                self.Log(f'..:: End Running {alName} Classifier ::..')

            self.Log(
                f'..:: End Running ML_Algoritms on "{file_name}" '
                f'for {target_column} '
                f'(random_state: {random_state}, '
                f'total features: {len(selected_features)}, ' # type: ignore
                f'preprocessing_method: {preprocessing_method}) ::..'
            )

        self.save_grouped_results(result_path)  # type: ignore

    # ------------------------------------------------------------------

    def round_percentage(self, number):
        return np.round(number * 100, 2)

    # ------------------------------------------------------------------

    def save_grouped_results(self, outPut_path):
        self.df_results = pd.DataFrame.from_dict(self.results, orient="index")
        self.df_results.index.name = "Algorithm"
        self.df_results.to_csv(f'{outPut_path}/results_df.csv')

        self.plot_groupScores(outPut_path)

    # ------------------------------------------------------------------

    def plot_groupScores(self, outPut_path):
        """
        Bar chart comparing all algorithms across every score metric.
        (Replaces the previous line-chart-per-feature-count plot, which
        required a ranked-features axis that no longer exists.)
        """
        mpl.rcParams.update({
            'font.family':       'serif',
            'font.serif':        ['Times New Roman'],
            'font.size':         12,
            'font.weight':       'normal',
            'axes.labelsize':    12,
            'axes.titlesize':    14,
            'axes.titleweight':  'bold',
            'xtick.labelsize':   10,
            'ytick.labelsize':   10,
            'legend.fontsize':   10,
        })

        # Collect mean scores for every algorithm × metric
        score_mean_cols = [s for s in self.scNames if s.endswith('_mean')]
        df_plot = self.df_results[score_mean_cols].copy()
        df_plot.columns = [
            c.replace('_mean', '').replace('_', ' ').upper()
            for c in score_mean_cols
        ]

        n_metrics = len(df_plot.columns)
        n_algos   = len(df_plot)
        x         = np.arange(n_metrics)
        bar_width  = 0.8 / n_algos

        fig, ax = plt.subplots(figsize=(max(14, n_metrics * 2), 7))

        for i, (algo, row) in enumerate(df_plot.iterrows()):
            offset = (i - n_algos / 2) * bar_width + bar_width / 2
            ax.bar(x + offset, row.values, bar_width, label=algo) # type: ignore

        ax.set_xticks(x)
        ax.set_xticklabels(df_plot.columns, rotation=30, ha='right')
        ax.set_ylabel('Score (%)')
        ax.set_title('Algorithm Comparison — All Metrics')
        ax.set_ylim(0, 110)
        ax.legend(loc='upper right')
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(f'{outPut_path}/algorithm_comparison.png', dpi=120)
        plt.close()

    # ------------------------------------------------------------------

    def plot_algorithmResults(self, alName, all_y_test, all_y_proba, outPut_path):

        mpl.rcParams.update({
            'font.family':       'serif',
            'font.serif':        ['Times New Roman'],
            'font.size':         12,
            'font.weight':       'normal',
            'axes.labelsize':    12,
            'axes.titlesize':    14,
            'axes.titleweight':  'bold',
            'xtick.labelsize':   10,
            'ytick.labelsize':   10,
            'legend.fontsize':   10,
        })

        fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 6))

        ax_roc.set_title("ROC Curve")
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')

        ax_pr.set_title("Precision-Recall Curve")
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')

        ax_roc.plot([0, 1], [0, 1], linestyle='--', label='No Skill', color='black')

        y        = np.concatenate(all_y_test)
        no_skill = len(y[y == 1]) / len(y)
        ax_pr.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill', color='black')

        tprs      = []
        roc_aucs  = []
        mean_fpr  = np.linspace(0, 1, 100)

        precisions  = []
        pr_aucs     = []
        mean_recall = np.linspace(0, 1, 100)

        for i in range(len(all_y_test)):
            y_test  = all_y_test[i]
            y_proba = all_y_proba[i]

            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            roc_auc     = auc(fpr, tpr)
            ax_roc.plot(
                fpr, tpr, alpha=0.3, linewidth=1,
                label=f'Fold {i+1} (AUC={self.round_percentage(roc_auc)}%)'
            )
            interp_tpr     = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0]  = 0.0
            tprs.append(interp_tpr)
            roc_aucs.append(roc_auc)

            precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])
            pr_auc               = auc(recall, precision)
            ax_pr.plot(
                recall, precision, alpha=0.3, linewidth=1,
                label=f'Fold {i+1} (AUC={self.round_percentage(pr_auc)}%)'
            )
            r_recall           = np.fliplr([recall])[0]
            r_precision        = np.fliplr([precision])[0]
            interp_precisions  = np.interp(mean_recall, r_recall, r_precision)
            interp_precisions[-1] = no_skill
            precisions.append(interp_precisions)
            pr_aucs.append(pr_auc)

        mean_tpr      = np.mean(tprs, axis=0)
        mean_tpr[-1]  = 1.0
        mean_roc_auc  = np.mean(roc_aucs, axis=0)
        std_roc_auc   = np.std(roc_aucs, axis=0)
        ax_roc.plot(
            mean_fpr, mean_tpr, color="b", lw=2,
            label=f'Mean (AUC={self.round_percentage(mean_roc_auc)}% '
                  f'\u00B1 {self.round_percentage(std_roc_auc)})'
        )
        std_tpr      = np.std(tprs, axis=0)
        tprs_upper   = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower   = np.maximum(mean_tpr - std_tpr, 0)
        ax_roc.fill_between(
            mean_fpr, tprs_lower, tprs_upper,
            color="grey", alpha=0.1, label='\u00B1 1 std. dev.'
        )

        mean_precision     = np.mean(precisions, axis=0)
        mean_precision[0]  = 1.0
        mean_pr_auc        = np.mean(pr_aucs, axis=0)
        std_pr_auc         = np.std(pr_aucs, axis=0)
        ax_pr.plot(
            mean_recall, mean_precision, color="b", lw=2,
            label=f'Mean (AUC={self.round_percentage(mean_pr_auc)}% '
                  f'\u00B1 {self.round_percentage(std_pr_auc)})'
        )
        std_precision      = np.std(precisions, axis=0)
        precisions_upper   = np.minimum(mean_precision + std_precision, 1)
        precisions_lower   = np.maximum(mean_precision - std_precision, 0)
        ax_pr.fill_between(
            mean_recall, precisions_lower, precisions_upper,
            color="grey", alpha=0.1, label='\u00B1 1 std. dev.'
        )

        ax_roc.grid(True)
        ax_pr.grid(True)
        ax_roc.legend(loc='best')
        ax_pr.legend(loc='best')

        plt.tight_layout()
        plt.savefig(f'{outPut_path}/by_Algorithms/{alName}.png', dpi=120)
        plt.close()

    # ------------------------------------------------------------------

    def get_algorithmLog(self, alName):
        results = self.results[alName]   # flat dict now — no n_features key
        log_str  = f'\n..:: {alName} Classifier Mean Results ::..\n'

        log_str += (
            f'==> Cross Validation Accuracy: '
            f'{results["cross_validation_accuracy_mean"]}%'
            f' \u00B1 {results["cross_validation_accuracy_std"]}\n'
        )
        log_str += (
            f'==> Test Accuracy: '
            f'{results["test_accuracy_mean"]}%'
            f' \u00B1 {results["test_accuracy_std"]}\n'
        )
        log_str += (
            f'==> Specificity: '
            f'{results["specificity_mean"]}%'
            f' \u00B1 {results["specificity_std"]}\n'
        )
        log_str += (
            f'==> Sensitivity: '
            f'{results["sensitivity_mean"]}%'
            f' \u00B1 {results["sensitivity_std"]}\n'
        )
        log_str += (
            f'==> MCC: '
            f'{results["mcc_mean"]}%'
            f' \u00B1 {results["mcc_std"]}\n'
        )
        log_str += (
            f'==> Precision: '
            f'{results["precision_mean"]}%'
            f' \u00B1 {results["precision_std"]}\n'
        )
        log_str += (
            f'==> Recall: '
            f'{results["recall_mean"]}%'
            f' \u00B1 {results["recall_std"]}\n'
        )
        log_str += (
            f'==> F1: '
            f'{results["f1_mean"]}%'
            f' \u00B1 {results["f1_std"]}\n'
        )
        log_str += (
            f'==> ROC-AUC: '
            f'{results["auc_roc_mean"]}%'
            f' \u00B1 {results["auc_roc_std"]}\n'
        )
        log_str += (
            f'==> AUPR: '
            f'{results["aupr_mean"]}%'
            f' \u00B1 {results["aupr_std"]}\n'
        )
        return log_str


    def calc_results(
            self, alName, all_y_test, all_y_pred, all_y_proba,
            all_cross_validation_accuracy, all_bestParams
    ):
        all_test_accuracy = []
        all_f1            = []
        all_precision     = []
        all_recall        = []
        all_auc_roc       = []
        all_aupr          = []
        all_specificity   = []
        all_sensitivity   = []
        all_mcc           = []

        for i in range(len(all_y_test)):
            y_test  = all_y_test[i]
            y_pred  = all_y_pred[i]
            y_proba = all_y_proba[i]

            all_test_accuracy.append(accuracy_score(y_test, y_pred))
            all_f1.append(f1_score(y_test, y_pred))

            all_precision.append(precision_score(y_test, y_pred, zero_division=0))
            all_recall.append(recall_score(y_test, y_pred, zero_division=0))

            if len(np.unique(y_test)) > 1:
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                all_auc_roc.append(auc(fpr, tpr))
            else:
                all_auc_roc.append(np.nan)

            if len(np.unique(y_test)) > 1:
                precision_c, recall_c, _ = precision_recall_curve(y_test, y_proba[:, 1])
                all_aupr.append(auc(recall_c, precision_c))
            else:
                all_aupr.append(np.nan)

            if len(np.unique(y_test)) > 1:
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                all_specificity.append(
                    np.round(tn / (fp + tn), 4) if (fp + tn) > 0 else np.nan
                )
                all_sensitivity.append(
                    np.round(tp / (tp + fn), 4) if (tp + fn) > 0 else np.nan
                )
            else:
                all_specificity.append(np.nan)
                all_sensitivity.append(np.nan)

            all_mcc.append(
                matthews_corrcoef(y_test, y_pred)
                if len(np.unique(y_test)) > 1 else np.nan
            )

        # Flat results dict — no n_features nesting
        self.results[alName] = {
            "cross_validation_accuracy_mean": self.round_percentage(np.nanmean(all_cross_validation_accuracy)),
            "cross_validation_accuracy_std":  self.round_percentage(np.nanstd(all_cross_validation_accuracy)),

            "test_accuracy_mean": self.round_percentage(np.nanmean(all_test_accuracy)),
            "test_accuracy_std":  self.round_percentage(np.nanstd(all_test_accuracy)),

            "specificity_mean": self.round_percentage(np.nanmean(all_specificity)),
            "specificity_std":  self.round_percentage(np.nanstd(all_specificity)),

            "sensitivity_mean": self.round_percentage(np.nanmean(all_sensitivity)),
            "sensitivity_std":  self.round_percentage(np.nanstd(all_sensitivity)),

            "mcc_mean": self.round_percentage(np.nanmean(all_mcc)),
            "mcc_std":  self.round_percentage(np.nanstd(all_mcc)),

            "precision_mean": self.round_percentage(np.nanmean(all_precision)),
            "precision_std":  self.round_percentage(np.nanstd(all_precision)),

            "recall_mean": self.round_percentage(np.nanmean(all_recall)),
            "recall_std":  self.round_percentage(np.nanstd(all_recall)),

            "f1_mean": self.round_percentage(np.nanmean(all_f1)),
            "f1_std":  self.round_percentage(np.nanstd(all_f1)),

            "auc_roc_mean": self.round_percentage(np.nanmean(all_auc_roc)),
            "auc_roc_std":  self.round_percentage(np.nanstd(all_auc_roc)),

            "aupr_mean": self.round_percentage(np.nanmean(all_aupr)),
            "aupr_std":  self.round_percentage(np.nanstd(all_aupr)),

            "best_param_fold_1": all_bestParams[0],
            "best_param_fold_2": all_bestParams[1],
            "best_param_fold_3": all_bestParams[2],
            "best_param_fold_4": all_bestParams[3],
            "best_param_fold_5": all_bestParams[4],
        }

    # ------------------------------------------------------------------

    def create_outPutFolder(self, tm_path):
        if not os.path.isdir(tm_path):
            os.makedirs(tm_path)
            os.makedirs(f'{tm_path}/by_Algorithms')
        for alName in self.alNames:
            tmp_path = f'{tm_path}/by_Algorithms/{alName}/'
            if not os.path.isdir(tmp_path):
                os.makedirs(tmp_path)