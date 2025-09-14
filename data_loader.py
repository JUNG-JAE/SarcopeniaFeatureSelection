import pyreadstat
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE
from feature_config import all_features, cate_nomial_features


class SarcoDataLoader:
    def __init__(self, data_path='./data/post-ref4.sav', test_size=0.2, random_state=42, batch_size=32, shuffle=True, k_fold=0, class_num=2, feature_indices=None):
        # Load data
        self.df, _ = pyreadstat.read_sav(data_path)
        no_missing_data = self.df.dropna(subset=all_features + ['Sarcopenia'])
        
        # One-hot encoding for categorical nominal features # for Random forest
        # X_df = no_missing_data[all_features].copy() #
        # for col in cate_nomial_features:
        #     if col in X_df.columns:
        #         X_df = pd.get_dummies(X_df, columns=[col], dtype=float)
        # self.X = X_df.values
        
        if feature_indices is not None:
            selected_features = [all_features[i] for i in feature_indices if i < len(all_features)]
        else:
            selected_features = all_features

        self.X = no_missing_data[selected_features].values
        
        if class_num == 2:
            self.y = no_missing_data['saecopenia_2'].astype(int).values
        elif class_num == 4:
            self.y = no_missing_data['Sarcopenia'].astype(int).values
               
        # Options
        self.test_size = test_size
        self.random_state = random_state
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.k_fold = k_fold
        self.sm = SMOTE(random_state=random_state)

    def _batch_generator(self, X, y, batch_size, shuffle=True):
        idxs = np.arange(len(X))
        if shuffle:
            np.random.shuffle(idxs)
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            batch_idx = idxs[start:end]
            yield X[batch_idx], y[batch_idx]

    def get_data_loader(self):
        # Standard Train/Test split
        if self.k_fold <= 1:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=self.y
            )

            X_train_res, y_train_res = self.sm.fit_resample(X_train, y_train) # SMOTE (Train only)

            train_loader = self._batch_generator(X_train_res, y_train_res, self.batch_size, self.shuffle)
            test_loader = self._batch_generator(X_test, y_test, self.batch_size, False)

            return train_loader, test_loader

        # K-Fold Cross Validation
        else:
            kf = StratifiedKFold(n_splits=self.k_fold, shuffle=True, random_state=self.random_state)
            folds = []

            for train_idx, test_idx in kf.split(self.X, self.y):
                X_train, X_test = self.X[train_idx], self.X[test_idx]
                y_train, y_test = self.y[train_idx], self.y[test_idx]

                X_train_res, y_train_res = self.sm.fit_resample(X_train, y_train) # SMOTE (Train only)

                train_loader = self._batch_generator(X_train_res, y_train_res, self.batch_size, self.shuffle)
                test_loader = self._batch_generator(X_test, y_test, self.batch_size, False)

                folds.append((train_loader, test_loader))

            return folds
