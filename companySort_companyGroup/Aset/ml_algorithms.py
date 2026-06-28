# ml_algorithms.py
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
    auc, matthews_corrcoef
)
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.metrics import make_scorer
from joblib import dump
from typing import Optional

from Aset.log import append_log

# Scoring multiple untuk CV
scoring_dict = {
    'accuracy':  'accuracy',
    'precision': make_scorer(precision_score, zero_division=0),
    'recall':    make_scorer(recall_score,    zero_division=0),
    'f1':        make_scorer(f1_score,        zero_division=0),
}


def run_hyperparameter_tuning(estimator, param_grid, scoring, kf, X_train, y_train,
                              sample_weights=None, groups=None, n_jobs=-1):
    grid_obj = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring_dict,   # multiple metrics
        refit='accuracy',       # pilih best berdasarkan accuracy
        cv=kf,
        n_jobs=n_jobs,
        error_score='raise'
    )

    fit_params = {}
    if sample_weights is not None:
        fit_params['sample_weight'] = sample_weights
    if groups is not None:
        fit_params['groups'] = groups

    grid_obj.fit(X_train, y_train, **fit_params)
    return grid_obj


def round_percentage(number):
    return np.round(number * 100, 2)


def create_output_folders(output_path: str, algorithms: list):
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "by_Algorithms"), exist_ok=True)
    for alName in algorithms:
        os.makedirs(os.path.join(output_path, "by_Algorithms", alName), exist_ok=True)


def prepare_data_for_fold(df_train: pd.DataFrame, df_test: pd.DataFrame, target_column: str):
    selected_features = [col for col in df_train.columns if col != target_column]
    X_train = df_train[selected_features]
    X_test  = df_test[selected_features]
    y_train = df_train[target_column]
    y_test  = df_test[target_column]
    return X_train, X_test, y_train, y_test, selected_features


def calculate_fold_metrics(y_test, y_pred, y_proba):
    metrics = {}
    metrics['accuracy']  = accuracy_score(y_test, y_pred)
    metrics['f1']        = f1_score(y_test, y_pred, zero_division=0)
    metrics['precision'] = precision_score(y_test, y_pred, zero_division=0)
    metrics['recall']    = recall_score(y_test, y_pred, zero_division=0)
    metrics['mcc']       = matthews_corrcoef(y_test, y_pred) if len(np.unique(y_test)) > 1 else np.nan

    if len(np.unique(y_test)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        metrics['specificity'] = tn / (fp + tn) if (fp + tn) > 0 else np.nan
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else np.nan

        fpr, tpr, _              = roc_curve(y_test, y_proba[:, 1])
        metrics['auc_roc']       = auc(fpr, tpr)
        precision_c, recall_c, _ = precision_recall_curve(y_test, y_proba[:, 1])
        metrics['aupr']          = auc(recall_c, precision_c)
    else:
        metrics['specificity'] = metrics['sensitivity'] = \
            metrics['auc_roc'] = metrics['aupr'] = np.nan

    return metrics


def compute_final_results(all_metrics, all_cv_scores, all_best_params, all_grid_objs):
    """
    all_grid_objs : list of fitted GridSearchCV objects, one per fold.
                    Needed to extract per-fold cv_results_ for CV metrics.
    """
    results = {}

    metric_names = ['accuracy', 'f1', 'precision', 'recall',
                    'specificity', 'sensitivity', 'mcc', 'auc_roc', 'aupr']

    for m in metric_names:
        values = [fold[m] for fold in all_metrics]
        results[f"{m}_mean"] = round_percentage(np.nanmean(values))
        results[f"{m}_std"]  = round_percentage(np.nanstd(values))

    for i, params in enumerate(all_best_params, 1):
        results[f"best_param_fold_{i}"] = params

    # Cross-validation metrics — aggregate cv_results_ across all fold grid objects
    # Each grid_obj.cv_results_['mean_test_X'] has one value per param combination;
    # we concatenate across folds so mean/std reflect the full search space.
    cv_acc       = np.concatenate([g.cv_results_['mean_test_accuracy']  for g in all_grid_objs])
    cv_precision = np.concatenate([g.cv_results_['mean_test_precision'] for g in all_grid_objs])
    cv_recall    = np.concatenate([g.cv_results_['mean_test_recall']    for g in all_grid_objs])
    cv_f1        = np.concatenate([g.cv_results_['mean_test_f1']        for g in all_grid_objs])

    results["cross_validation_accuracy_mean"]  = round_percentage(np.nanmean(cv_acc))
    results["cross_validation_accuracy_std"]   = round_percentage(np.nanstd(cv_acc))

    results["cross_validation_precision_mean"] = round_percentage(np.nanmean(cv_precision))
    results["cross_validation_precision_std"]  = round_percentage(np.nanstd(cv_precision))

    results["cross_validation_recall_mean"]    = round_percentage(np.nanmean(cv_recall))
    results["cross_validation_recall_std"]     = round_percentage(np.nanstd(cv_recall))

    results["cross_validation_f1_mean"]        = round_percentage(np.nanmean(cv_f1))
    results["cross_validation_f1_std"]         = round_percentage(np.nanstd(cv_f1))

    return results


def save_detailed_results(output_path: str, all_results: dict):
    df = pd.DataFrame.from_dict(all_results, orient="index")
    df.index.name = "Algorithm"

    important_cols = [
        'accuracy_mean', 'accuracy_std',
        'precision_mean', 'precision_std',
        'recall_mean', 'recall_std',
        'f1_mean', 'f1_std',
        'auc_roc_mean', 'aupr_mean',
    ]
    existing = [c for c in important_cols if c in df.columns]
    others   = [c for c in df.columns if c not in existing]
    df = df[existing + others]

    csv_path = os.path.join(output_path, "results_metrics.csv")
    df.to_csv(csv_path)
    print(f"Metrik disimpan → {csv_path}")
    return df


def plot_test_metrics_comparison(df_results: pd.DataFrame, output_path: str):
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12, 'axes.titleweight': 'bold'})

    metrics_to_plot = ['accuracy_mean', 'precision_mean', 'recall_mean', 'f1_mean']
    metric_names    = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    plot_data = df_results[[c for c in metrics_to_plot if c in df_results.columns]].copy()
    plot_data.columns = metric_names[:len(plot_data.columns)]

    fig, ax = plt.subplots(figsize=(12, 7))
    x         = np.arange(len(plot_data.columns))
    bar_width = 0.8 / len(plot_data)

    for i, (algo, row) in enumerate(plot_data.iterrows()):
        offset = (i - len(plot_data) / 2) * bar_width + bar_width / 2
        ax.bar(x + offset, row.values, bar_width, label=algo) # type: ignore[reportUnknownMemberType]

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel('Score')
    ax.set_title('Perbandingan Metrik Test')
    ax.set_ylim(0, 1.05)
    ax.legend(title="Algorithm", loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "test_metrics_comparison.png"), dpi=200)
    plt.close()


def plot_group_scores(results_df: pd.DataFrame, output_path: str):
    pass  # placeholder — isi sesuai kebutuhan


def get_algorithm_log(alName, results):
    log_str  = f'\n..:: {alName} Classifier Mean Results ::..\n'
    log_str += f'==> Cross Validation Accuracy : {results["cross_validation_accuracy_mean"]}% ± {results["cross_validation_accuracy_std"]}\n'
    log_str += f'==> Test Accuracy             : {results["accuracy_mean"]}% ± {results["accuracy_std"]}\n'
    log_str += f'==> F1                        : {results["f1_mean"]}% ± {results["f1_std"]}\n'
    log_str += f'==> Precision                 : {results["precision_mean"]}% ± {results["precision_std"]}\n'
    log_str += f'==> Recall                    : {results["recall_mean"]}% ± {results["recall_std"]}\n'
    log_str += f'==> ROC-AUC                   : {results["auc_roc_mean"]}% ± {results["auc_roc_std"]}\n'
    log_str += f'==> AUPR                      : {results["aupr_mean"]}% ± {results["aupr_std"]}\n'
    return log_str


def plot_algorithm_results(alName: str, all_y_test, all_y_proba, output_path: str):
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'font.size': 12, 'axes.titlesize': 14, 'axes.titleweight': 'bold'
    })

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))

    ax_roc.set_title("ROC Curve")
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.plot([0, 1], [0, 1], 'k--', label='No Skill')

    ax_pr.set_title("Precision-Recall Curve")
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')

    y_all    = np.concatenate(all_y_test)
    no_skill = len(y_all[y_all == 1]) / len(y_all)
    ax_pr.plot([0, 1], [no_skill, no_skill], 'k--', label='No Skill')

    mean_fpr    = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)
    tprs, precisions = [], []
    roc_aucs, pr_aucs = [], []

    for i in range(len(all_y_test)):
        y_test  = all_y_test[i]
        y_proba = all_y_proba[i]

        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        roc_auc     = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, alpha=0.3, lw=1,
                    label=f'Fold {i+1} (AUC={round_percentage(roc_auc)}%)')
        interp_tpr    = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        roc_aucs.append(roc_auc)

        precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])
        pr_auc               = auc(recall, precision)
        ax_pr.plot(recall, precision, alpha=0.3, lw=1,
                   label=f'Fold {i+1} (AUC={round_percentage(pr_auc)}%)')
        r_recall      = np.fliplr([recall])[0]
        r_precision   = np.fliplr([precision])[0]
        interp_prec   = np.interp(mean_recall, r_recall, r_precision)
        interp_prec[-1] = no_skill
        precisions.append(interp_prec)
        pr_aucs.append(pr_auc)

    mean_tpr      = np.mean(tprs, axis=0)
    mean_tpr[-1]  = 1.0
    mean_roc      = np.mean(roc_aucs)
    std_roc       = np.std(roc_aucs)
    ax_roc.plot(mean_fpr, mean_tpr, 'b', lw=2,
                label=f'Mean (AUC={round_percentage(mean_roc)}% ± {round_percentage(std_roc)})')
    std_tpr    = np.std(tprs, axis=0)
    ax_roc.fill_between(mean_fpr,
                        np.maximum(mean_tpr - std_tpr, 0),
                        np.minimum(mean_tpr + std_tpr, 1),
                        color='grey', alpha=0.1, label='± 1 std. dev.')

    mean_prec     = np.mean(precisions, axis=0)
    mean_prec[0]  = 1.0
    mean_pr       = np.mean(pr_aucs)
    std_pr        = np.std(pr_aucs)
    ax_pr.plot(mean_recall, mean_prec, 'b', lw=2,
               label=f'Mean (AUC={round_percentage(mean_pr)}% ± {round_percentage(std_pr)})')
    std_prec = np.std(precisions, axis=0)
    ax_pr.fill_between(mean_recall,
                       np.maximum(mean_prec - std_prec, 0),
                       np.minimum(mean_prec + std_prec, 1),
                       color='grey', alpha=0.1, label='± 1 std. dev.')

    ax_roc.legend(loc='lower right')
    ax_pr.legend(loc='lower left')
    ax_roc.grid(True, alpha=0.3)
    ax_pr.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, "by_Algorithms", alName, f"{alName}_curves.png"),
        dpi=150
    )
    plt.close()


def run_ml_algorithms(
    input_path: str,
    output_path: str,
    file_name: str,
    target_columns: list,
    algorithms: dict,
    scorer: str,
    n_jobs: int = -1,
    preprocessing_method: str = "Standard",
    random_state: int = 42,
    number_of_splits: int = 5,
    kf=None,
    group_column: Optional[str] = None,
    save_models: bool = True,
    log_path: Optional[str] = None,
    log_filename: str = "app_log"
):
    if log_path is None:
        log_path = output_path

    alNames = list(algorithms.keys())

    for target_column in target_columns:
        append_log(log_path, log_filename, f'..:: Start ML for {target_column} ::..')

        result_path = os.path.join(
            output_path, target_column, f"{random_state}_{preprocessing_method}"
        )
        create_output_folders(result_path, alNames)

        all_results = {}

        for alName in alNames:
            append_log(log_path, log_filename, f'..:: Running {alName} ::..')

            alg_config = algorithms[alName]
            alg_config["param_grid"]['random_state'] = [random_state]

            all_metrics    = []
            all_cv_scores  = []
            all_best_params = []
            all_grid_objs  = []   # ← kumpulkan semua grid_obj per fold
            all_y_test     = []
            all_y_proba    = []

            for fold_number in range(1, number_of_splits + 1):
                train_path = os.path.join(
                    input_path, target_column,
                    f"{random_state}_{preprocessing_method}",
                    "train", f"fold_{fold_number}.csv"
                )
                test_path = os.path.join(
                    input_path, target_column,
                    f"{random_state}_{preprocessing_method}",
                    "test", f"fold_{fold_number}.csv"
                )

                df_train = pd.read_csv(train_path)
                df_test  = pd.read_csv(test_path)

                X_train, X_test, y_train, y_test, _ = prepare_data_for_fold(
                    df_train, df_test, target_column
                )

                if alName != "HGB":
                    cw = compute_class_weight("balanced",
                                             classes=np.unique(y_train), y=y_train)
                    class_weight_dict = dict(zip(np.unique(y_train), cw))
                    alg_config["param_grid"]['class_weight'] = [class_weight_dict]
                    sample_weights = compute_sample_weight(class_weight_dict, y_train)
                else:
                    sample_weights = None

                groups = df_train[group_column] if group_column is not None else None

                grid_obj = run_hyperparameter_tuning(
                    alg_config["estimator"], alg_config["param_grid"],
                    scorer, kf, X_train, y_train,
                    sample_weights, groups, n_jobs
                )

                best_est = grid_obj.best_estimator_
                y_pred   = best_est.predict(X_test)
                y_proba  = best_est.predict_proba(X_test)

                if save_models:
                    dump(
                        best_est,
                        os.path.join(result_path, "by_Algorithms", alName,
                                     f"fold{fold_number}.joblib")
                    )

                fold_metrics = calculate_fold_metrics(y_test, y_pred, y_proba)
                all_metrics.append(fold_metrics)
                all_cv_scores.append(grid_obj.best_score_)
                all_best_params.append(grid_obj.best_params_)
                all_grid_objs.append(grid_obj)   # ← simpan grid_obj
                all_y_test.append(y_test)
                all_y_proba.append(y_proba)

            # Teruskan all_grid_objs ke compute_final_results
            final_results = compute_final_results(
                all_metrics, all_cv_scores, all_best_params, all_grid_objs
            )
            all_results[alName] = final_results

            append_log(log_path, log_filename, get_algorithm_log(alName, final_results))
            plot_algorithm_results(alName, all_y_test, all_y_proba, result_path)

        df_results = save_detailed_results(result_path, all_results)
        plot_test_metrics_comparison(df_results, result_path)
        plot_group_scores(df_results, result_path)

        append_log(log_path, log_filename, f'..:: End ML for {target_column} ::..')

    print(f"ML Training selesai untuk {file_name}")