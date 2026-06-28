# fp_augmentation.py
import os
import pandas as pd
import numpy as np
from typing import Optional

from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

from Aset.log import append_log


def to_transactions(X: pd.DataFrame, feature_cols: list) -> list:
    """Convert DataFrame ke list of transactions (value > 0 dianggap ada)"""
    transactions = []
    for _, row in X.iterrows():
        items = [col for col in feature_cols if row[col] > 0]
        transactions.append(items if items else ['__empty__'])
    return transactions


def mine_frequent_itemsets(df_te: pd.DataFrame, 
                          method: str, 
                          min_support: float, 
                          max_itemset_len: Optional[int] = None):
    """Mining frequent itemsets menggunakan Apriori atau FP-Growth"""
    kwargs = {
        'df': df_te,
        'min_support': min_support,
        'use_colnames': True
    }
    if max_itemset_len is not None:
        kwargs['max_len'] = max_itemset_len

    if method == 'apriori':
        return apriori(**kwargs)
    elif method == 'fpgrowth':
        return fpgrowth(**kwargs)
    else:
        raise ValueError(f"Unknown mining_method: {method}. Gunakan 'apriori' atau 'fpgrowth'.")


def mine_association_rules(frequent_itemsets: pd.DataFrame, min_confidence: float):
    """Generate association rules"""
    if frequent_itemsets.empty:
        return pd.DataFrame()
    try:
        return association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
    except:
        return pd.DataFrame()


def generate_itemset_features(frequent_itemsets: pd.DataFrame, 
                             X_train: pd.DataFrame, 
                             X_test: pd.DataFrame):
    """Buat binary features dari frequent itemsets"""
    if frequent_itemsets.empty:
        return {'train': pd.DataFrame(), 'test': pd.DataFrame()}

    itemsets = frequent_itemsets['itemsets'].tolist()

    def _encode(X: pd.DataFrame) -> pd.DataFrame:
        cols = {}
        for idx, itemset in enumerate(itemsets):
            col_name = f'fp_{idx}'
            valid_items = [it for it in itemset if it in X.columns]
            if not valid_items:
                cols[col_name] = np.zeros(len(X), dtype=int)
            else:
                cols[col_name] = X[valid_items].gt(0).all(axis=1).astype(int).values
        return pd.DataFrame(cols, index=X.index)

    return {
        'train': _encode(X_train),
        'test': _encode(X_test)
    }


def generate_rule_features(rules: pd.DataFrame, 
                          X_train: pd.DataFrame, 
                          X_test: pd.DataFrame):
    """Buat binary features dari association rules (antecedents)"""
    if rules.empty:
        return {'train': pd.DataFrame(), 'test': pd.DataFrame()}

    antecedents = rules['antecedents'].tolist()

    def _encode(X: pd.DataFrame) -> pd.DataFrame:
        cols = {}
        for idx, antecedent in enumerate(antecedents):
            col_name = f'rule_{idx}'
            valid_items = [it for it in antecedent if it in X.columns]
            if not valid_items:
                cols[col_name] = np.zeros(len(X), dtype=int)
            else:
                cols[col_name] = X[valid_items].gt(0).all(axis=1).astype(int).values
        return pd.DataFrame(cols, index=X.index)

    return {
        'train': _encode(X_train),
        'test': _encode(X_test)
    }


def run_fp_augmentation(
    input_path: str,
    output_path: str,
    file_name: str,
    target_columns: list,
    mining_method: str = "fpgrowth",      # "apriori" atau "fpgrowth"
    augment_mode: str = "both",           # "itemsets" | "rules" | "both"
    min_support: float = 0.1,
    min_confidence: float = 0.7,
    max_itemset_len: Optional[int] = None,
    preprocessing_method: str = "Standard",
    random_state: int = 42,
    number_of_splits: int = 5,
    log_path: Optional[str] = None,
    log_filename: str = "app_log"
):
    """
    Fungsi utama FP Augmentation
    """
    if log_path is None:
        log_path = output_path

    for target_column in target_columns:
        fold_base = os.path.join(
            input_path, target_column, f"{random_state}_{preprocessing_method}"
        )

        append_log(
            log_path, log_filename,
            f'..:: Start FP_Augmentation on "{file_name}" for {target_column} '
            f'(mining: {mining_method}, mode: {augment_mode}, '
            f'min_support: {min_support}) ::..'
        )

        for fold_number in range(1, number_of_splits + 1):
            append_log(log_path, log_filename, f'  -> Processing fold {fold_number} ...')

            train_path = os.path.join(fold_base, "train", f"fold_{fold_number}.csv")
            test_path  = os.path.join(fold_base, "test",  f"fold_{fold_number}.csv")

            df_train = pd.read_csv(train_path)
            df_test  = pd.read_csv(test_path)

            # Feature columns (hindari kolom target dan augmentasi sebelumnya)
            non_feature_cols = [target_column] + [
                c for c in df_train.columns if c.startswith(('fp_', 'rule_'))
            ]
            feature_cols = [c for c in df_train.columns if c not in non_feature_cols]

            X_train = df_train[feature_cols]
            X_test  = df_test[feature_cols]

            # Convert to transactions
            train_transactions = to_transactions(X_train, feature_cols)

            te = TransactionEncoder()
            te_array = te.fit(train_transactions).transform(train_transactions)
            df_te_train = pd.DataFrame(te_array, columns=te.columns_) # pyright: ignore[reportArgumentType]

            # Mining
            frequent_itemsets = mine_frequent_itemsets(
                df_te_train, mining_method, min_support, max_itemset_len
            )

            if frequent_itemsets.empty:
                append_log(log_path, log_filename, 
                          f'    [WARN] No frequent itemsets found for fold {fold_number}. Skipping.')
                continue

            append_log(log_path, log_filename, 
                      f'    Found {len(frequent_itemsets)} frequent itemset(s).')

            # Generate features
            fp_cols = {}
            rule_cols = {}

            if augment_mode in ('itemsets', 'both'):
                fp_cols = generate_itemset_features(frequent_itemsets, X_train, X_test)

            if augment_mode in ('rules', 'both'):
                rules = mine_association_rules(frequent_itemsets, min_confidence)
                if not rules.empty:
                    rule_cols = generate_rule_features(rules, X_train, X_test)
                    append_log(log_path, log_filename, 
                              f'    Generated {len(rule_cols["train"].columns)} rule feature(s).')
                else:
                    append_log(log_path, log_filename, 
                              f'    [WARN] No association rules found (min_confidence={min_confidence}).')

            # Augment data
            df_train_aug = df_train.copy()
            df_test_aug  = df_test.copy()

            for side, df_aug in [('train', df_train_aug), ('test', df_test_aug)]:
                if not fp_cols.get(side, pd.DataFrame()).empty:
                    df_aug = pd.concat([df_aug.reset_index(drop=True), 
                                      fp_cols[side].reset_index(drop=True)], axis=1)
                if not rule_cols.get(side, pd.DataFrame()).empty:
                    df_aug = pd.concat([df_aug.reset_index(drop=True), 
                                      rule_cols[side].reset_index(drop=True)], axis=1)

                # Simpan kembali
                if side == 'train':
                    df_train_aug = df_aug
                else:
                    df_test_aug = df_aug

            df_train_aug.to_csv(train_path, index=False)
            df_test_aug.to_csv(test_path, index=False)

            append_log(
                log_path, log_filename,
                f'    Fold {fold_number} augmented — '
                f'train: {df_train_aug.shape}, test: {df_test_aug.shape}'
            )

        append_log(
            log_path, log_filename,
            f'..:: End FP_Augmentation on "{file_name}" for {target_column} ::..'
        )

    print(f"FP Augmentation selesai untuk file: {file_name}")