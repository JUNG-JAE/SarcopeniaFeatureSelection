
import numpy as np
from catboost import CatBoostClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold

from feature_config import all_features
from data_loader import SarcoDataLoader


class FeatureSelectionEnv:
    def __init__(self, test_size=0.2, cat_iterations=200, cat_random_state=0, k_fold: int = 0, random_state: int = 42, run_dir: str = "./runs"):
        data = load_breast_cancer()
        X, y = data.data.astype(np.float32), data.target.astype(np.int64)
        self.feature_names = list(data.feature_names)
        self.n_features = X.shape[1]
        self.run_dir = f"{run_dir}/catboost_logs"

        # For K-fold, store the entire dataset
        self.X = X
        self.y = y
        
        # no k-fold
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y) # split train, test dataset
        self.cb_iterations = cat_iterations
        self.cb_random_state = cat_random_state
        
        # k-fold
        self.k_fold = int(k_fold) if k_fold is not None else 0
        self.random_state = random_state
        
    def _eval_subset_accuracy(self, mask: np.ndarray) -> float:
        # mask: [n_features] with 0/1
        if mask.sum() == 0:
            return 0.0  # No feature selection results in a reward of 0
        
        cols = np.where(mask == 1)[0]
        clf = CatBoostClassifier(iterations=self.cb_iterations, depth=6, learning_rate=0.1, loss_function="Logloss", 
                                 eval_metric="Accuracy", random_seed=self.cb_random_state, verbose=False, train_dir=self.run_dir)
        clf.fit(self.X_train[:, cols], self.y_train)
        pred = clf.predict(self.X_val[:, cols]).ravel()
        
        return float(accuracy_score(self.y_val, pred))
    
    def evaluate_accuracy(self, mask: np.ndarray) -> float:
        # Train/evaluate the CatBoost using the selected features (given feature mask).
        m = np.asarray(mask).astype(bool)
        if m.sum() == 0:
            return 0.0

        if self.k_fold and self.k_fold >= 2:
            skf = StratifiedKFold(n_splits=self.k_fold, shuffle=True, random_state=self.random_state)
            accs = []
            for tr_idx, va_idx in skf.split(self.X, self.y):
                X_tr, y_tr = self.X[tr_idx][:, m], self.y[tr_idx]
                X_va, y_va = self.X[va_idx][:, m], self.y[va_idx]
                if X_tr.size == 0 or X_va.size == 0:
                    continue

                clf = CatBoostClassifier(iterations=self.cb_iterations, depth=6, learning_rate=0.1, loss_function="Logloss", 
                                         eval_metric="Accuracy", random_seed=self.cb_random_state, verbose=False, train_dir=self.run_dir)
                clf.fit(X_tr, y_tr)
                pred = clf.predict(X_va).ravel()
                accs.append(float(accuracy_score(y_va, pred)))

            return float(np.mean(accs)) if accs else 0.0
        
        X_tr = self.X_train[:, m]
        y_tr = self.y_train
        X_va = self.X_val[:, m]
        y_va = self.y_val

        clf = CatBoostClassifier(iterations=self.cb_iterations, depth=6, learning_rate=0.1, loss_function="Logloss", 
                                 eval_metric="Accuracy", random_seed=self.cb_random_state, verbose=False, train_dir=self.run_dir)
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_va).ravel()
        
        return float(accuracy_score(y_va, pred))


class FeatureSelectionEnvSarcopenia(FeatureSelectionEnv):
    def __init__(self, data_path: str = "./data/post-ref4.sav",test_size: float = 0.2, cat_iterations: int = 200, cat_random_state: int = 0, 
                 k_fold: int = 0, random_state: int = 42, batch_size: int = 512, shuffle: bool = True, class_num = 2, run_dir: str = "./runs"):
        # The breast cancer dataset, it is not invoked
        self.cb_iterations = cat_iterations
        self.cb_random_state = cat_random_state
        self.run_dir = f"{run_dir}/catboost_logs"
        
        # k-fold
        self.k_fold = k_fold
        self.random_state = random_state
        self.test_size = test_size

        # num of class
        self.class_num = class_num
        
        # feature_config
        self.feature_names = list(all_features)
        self.n_features = len(self.feature_names)

        self._dl = SarcoDataLoader(data_path=data_path, test_size=test_size, random_state=random_state, batch_size=batch_size, shuffle=shuffle, k_fold=k_fold)
                
    @staticmethod
    def _concat_from_loader(loader):
        # Merge the batch generator from "data_loader" into into a single (X, y) ndarray.
        xs, ys = [], []
        for Xb, yb in loader:
            xs.append(Xb); ys.append(yb)
        
        X = np.vstack(xs) if xs else np.empty((0, 0))
        y = np.concatenate(ys) if ys else np.empty((0,), dtype=int)
        
        return X, y

    def evaluate_accuracy(self, mask):
        m = np.asarray(mask, dtype=bool)
        if m.sum() == 0:
            return 0.0

        if self.k_fold <= 1:
            # hold-out: (train_loader, test_loader)
            train_loader, test_loader = self._dl.get_data_loader()
            X_tr, y_tr = self._concat_from_loader(train_loader)
            X_te, y_te = self._concat_from_loader(test_loader)

            if X_tr.size == 0 or X_te.size == 0:
                return 0.0

            if self.class_num == 2:
                clf = CatBoostClassifier(iterations=self.cb_iterations, depth=6, learning_rate=0.1, loss_function="Logloss", 
                                        eval_metric="Accuracy", random_seed=self.cb_random_state, verbose=False, train_dir=self.run_dir)
            elif self.class_num == 4:
                clf = CatBoostClassifier(iterations=self.cb_iterations, depth=6, learning_rate=0.1, loss_function="MultiClass", 
                                        eval_metric="Accuracy", random_seed=self.cb_random_state, verbose=False, train_dir=self.run_dir)
            
            clf.fit(X_tr[:, m], y_tr)
            pred = clf.predict(X_te[:, m]).ravel()
            
            return float(accuracy_score(y_te, pred))

        else:
            folds = self._dl.get_data_loader()
            accs = []
            for train_loader, test_loader in folds:
                X_tr, y_tr = self._concat_from_loader(train_loader)
                X_te, y_te = self._concat_from_loader(test_loader)
                if X_tr.size == 0 or X_te.size == 0:
                    accs.append(0.0); continue

                if self.class_num == 2:
                    clf = CatBoostClassifier(iterations=self.cb_iterations, depth=6, learning_rate=0.1, loss_function="Logloss",
                                            eval_metric="Accuracy", random_seed=self.cb_random_state, verbose=False, train_dir=self.run_dir)
                elif self.class_num == 4:
                    clf = CatBoostClassifier(iterations=self.cb_iterations, depth=6, learning_rate=0.1, loss_function="MultiClass", 
                                            eval_metric="Accuracy", random_seed=self.cb_random_state, verbose=False, train_dir=self.run_dir)
                
                clf.fit(X_tr[:, m], y_tr)
                pred = clf.predict(X_te[:, m]).ravel()
                accs.append(accuracy_score(y_te, pred))
                
            return float(np.mean(accs)) if accs else 0.0


def _score_accuracy_with_mask(env, mask: np.ndarray) -> float:
    # Try several likely method names in FeatureSelectionEnv to get accuracy for a given feature mask.
    # Common method names tried in user's envs
    for name in ["evaluate_accuracy", "score_accuracy", "eval_accuracy", "score", "evaluate", "rf_accuracy"]:
        fn = getattr(env, name, None)
        if callable(fn):
            return float(fn(mask))
    # If none exists, raise a clear error so you can wire one line in FeatureSelectionEnv.
    raise AttributeError("FeatureSelectionEnv needs a method to score accuracy given a binary mask.")


def impartial_rewards_accuracy(env, A_t_prev: np.ndarray, A_t: np.ndarray, penalty: float) -> tuple[list[float], float, float]:
    F_t = _score_accuracy_with_mask(env, A_t)
    F_prev = _score_accuracy_with_mask(env, A_t_prev) if A_t_prev is not None else 0.0
    F_delta = F_t - F_prev # Current and previous performance

    # Agents that change actions in this episode
    baseline = A_t_prev if A_t_prev is not None else np.zeros_like(A_t)
    changed_idx = np.flatnonzero(A_t != baseline)

    n = len(A_t)
    rewards = [-penalty if A_t[i] == 1 else 0.0 for i in range(n)]
    
    # If no overall improvement, no rewards
    if F_delta == 0.0:
        return [0.0] * len(A_t), F_t, F_delta

    # Compute individual contributions (Delta_i) by flipping only agent
    deltas = np.zeros(n, dtype=float) # Default 0.0 (unchanged agents remain 0)
    for i in changed_idx:
        A_switch = A_t.copy()
        A_switch[i] = 1 - A_switch[i] # Flip only agent i
        F_indiv_i = _score_accuracy_with_mask(env, A_switch)
        deltas[i] = F_t - F_indiv_i

    aligned = [i for i in changed_idx if deltas[i] * F_delta > 0]

    if not aligned:
        return [0.0] * n, F_t, F_delta

    denom = sum(abs(deltas[i]) for i in aligned)

    if denom == 0.0:
        return [0.0] * n, F_t, F_delta
    
    # Reward distribution: aligned agents share F_delta proportionally to |Delta_i|
    rewards = [0.0] * n
    for i in aligned:
        rewards[i] = F_delta * (abs(deltas[i]) / denom) * 100

    return rewards, F_t, F_delta



