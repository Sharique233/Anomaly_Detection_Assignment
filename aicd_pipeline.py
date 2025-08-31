"""
AICD Anomaly Detection - Single Pipeline

Provides a single, importable and executable pipeline for:
- Loading & EDA helpers
- Preprocessing
- Training multiple anomaly models (Autoencoder, Isolation Forest, One-Class SVM, LOF, Statistical)
- Evaluation & comparison
- Saving/loading models
- Prediction (single model and ensemble)

CLI usage:
  python aicd_pipeline.py train --data_path authenticIndustrialCloudDataDataset/data
  python aicd_pipeline.py predict --input authenticIndustrialCloudDataDataset/data/Data4.csv --output predictions.csv
"""

import argparse
import json
import warnings
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

import joblib

warnings.filterwarnings("ignore")

if os.environ.get('AICD_DISABLE_TF', '') == '1':
    TF_AVAILABLE = False
else:
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.optimizers import Adam
        TF_AVAILABLE = True
    except Exception:
        TF_AVAILABLE = False


@dataclass
class PreprocessingInfo:
    removed_columns: list
    selected_features: list
    original_shape: Tuple[int, int]
    final_shape: Tuple[int, int]
    anomaly_rate: float
    train_anomaly_rate: float
    test_anomaly_rate: float


class AICDPipeline:
    def __init__(self, target_column: str = 'Alarm.ItemDroppedError', random_state: int = 42):
        self.target_column = target_column
        self.random_state = random_state
        self.scaler = None
        self.selected_features = []
        self.removed_columns = []
        self.models: Dict[str, Any] = {}
        self.model_performance: Dict[str, Dict[str, float]] = {}

    # ------------------ Data loading & EDA helpers ------------------
    def load_all_data(
        self,
        data_path: str,
        files: Optional[list] = None,
        sep: str = ';',
        max_rows: Optional[int] = None,
        chunksize: Optional[int] = None,
        usecols: Optional[list] = None,
        downcast_numeric: bool = True,
    ) -> pd.DataFrame:
        if files is None:
            files = ['Data1.csv', 'Data2.csv', 'Data3.csv', 'Data4.csv', 'Data5.csv']
        frames = []
        total_rows_collected = 0
        target_row_limit = max_rows if max_rows is not None and max_rows > 0 else None

        for f in files:
            p = Path(data_path) / f
            if not p.exists():
                continue
            # If no limits, read whole file at once
            if target_row_limit is None and chunksize is None:
                df = pd.read_csv(p, sep=sep, usecols=usecols)
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                frames.append(df)
                continue

            # Chunked reading to cap memory
            effective_chunksize = chunksize if chunksize is not None and chunksize > 0 else 100_000
            for chunk in pd.read_csv(p, sep=sep, usecols=usecols, chunksize=effective_chunksize):
                chunk = chunk.loc[:, ~chunk.columns.str.contains('^Unnamed')]
                if target_row_limit is not None:
                    remaining = target_row_limit - total_rows_collected
                    if remaining <= 0:
                        break
                    if len(chunk) > remaining:
                        chunk = chunk.head(remaining)
                frames.append(chunk)
                total_rows_collected += len(chunk)
                if target_row_limit is not None and total_rows_collected >= target_row_limit:
                    break

        if not frames:
            raise FileNotFoundError(f"No data files found in {data_path}")

        combined = pd.concat(frames, ignore_index=True)

        if downcast_numeric:
            combined = self._downcast_numeric(combined)
        return combined

    def _downcast_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Downcast numeric columns to reduce memory footprint."""
        result = df.copy()
        float_cols = result.select_dtypes(include=['float64']).columns
        int_cols = result.select_dtypes(include=['int64', 'int32']).columns
        if len(float_cols) > 0:
            result[float_cols] = result[float_cols].astype(np.float32)
        if len(int_cols) > 0:
            # Keep target as small int as well; sklearn handles it fine
            result[int_cols] = result[int_cols].astype(np.int32)
        return result

    def quick_eda(self, df: pd.DataFrame) -> Dict[str, Any]:
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_total': int(df.isnull().sum().sum()),
            'target_present': self.target_column in df.columns
        }
        if info['target_present']:
            vc = df[self.target_column].value_counts(dropna=False)
            info['target_distribution'] = vc.to_dict()
        return info

    # ------------------ Preprocessing ------------------
    def _remove_problematic_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_remove = [
            'Statistics.RT_VacuumBlower',
            'Statistics.RT_BellowBlower',
            'Completed.SeqNrPickData',
            'Completed.OnePickCompleted',
            'Date', 'Time', 'Relative time'
        ]
        existing = [c for c in cols_to_remove if c in df.columns]
        if existing:
            df = df.drop(columns=existing)
            self.removed_columns.extend(existing)
        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.isnull().sum().sum() == 0:
            return df
        num_cols = df.select_dtypes(include=[np.number]).columns
        imp = SimpleImputer(strategy='median')
        df[num_cols] = imp.fit_transform(df[num_cols])
        return df

    def _scale(self, X_train: pd.DataFrame, X_test: Optional[pd.DataFrame], scaler: str = 'standard'):
        if scaler == 'standard':
            self.scaler = StandardScaler()
        elif scaler == 'robust':
            self.scaler = RobustScaler()
        elif scaler == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Unknown scaler")
        X_tr = pd.DataFrame(self.scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        # Reduce dtype size to save memory
        X_tr = X_tr.astype(np.float32)
        X_te = None
        if X_test is not None:
            X_te = pd.DataFrame(self.scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
            X_te = X_te.astype(np.float32)
        return X_tr, X_te

    def _select_features(self, X: pd.DataFrame, y: pd.Series, k: int = 50):
        # To avoid huge temporary allocations, compute correlation on a row sample if needed
        max_rows_for_corr = 200_000
        if len(X) > max_rows_for_corr:
            sampled_idx = X.sample(n=max_rows_for_corr, random_state=self.random_state).index
            X_corr = X.loc[sampled_idx]
            y_corr = y.loc[sampled_idx]
        else:
            X_corr = X
            y_corr = y
        # Ensure float dtype to reduce memory and avoid integer upcasting
        X_corr = X_corr.astype(np.float32)
        y_corr = y_corr.astype(np.float32)
        corrs = X_corr.corrwith(y_corr).abs().sort_values(ascending=False)
        top = corrs.head(min(k, len(corrs))).index.tolist()
        self.selected_features = top
        return X[top]

    def preprocess(self, df: pd.DataFrame, test_size: float = 0.2, scaler: str = 'standard', n_features: int = 50) -> Dict[str, Any]:
        df = self._remove_problematic_columns(df.copy())
        df = self._handle_missing(df)
        if self.target_column not in df.columns:
            raise ValueError(f"Target column {self.target_column} not present")
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column].astype(np.int32)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        X_train, X_test = self._scale(X_train, X_test, scaler=scaler)
        X_train = self._select_features(X_train, y_train, k=min(n_features, X_train.shape[1]))
        X_test = X_test[self.selected_features]
        info = PreprocessingInfo(
            removed_columns=self.removed_columns,
            selected_features=self.selected_features,
            original_shape=df.shape,
            final_shape=X.shape,
            anomaly_rate=float(y.mean()),
            train_anomaly_rate=float(y_train.mean()),
            test_anomaly_rate=float(y_test.mean())
        )
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'info': info
        }

    # ------------------ Models ------------------
    def _create_autoencoder(self, input_dim: int, encoding_dim: int = 32, dropout: float = 0.2):
        if not TF_AVAILABLE:
            return None
        inp = Input(shape=(input_dim,))
        x = Dense(encoding_dim * 4, activation='relu')(inp)
        x = Dropout(dropout)(x)
        x = Dense(encoding_dim * 2, activation='relu')(x)
        x = Dropout(dropout)(x)
        code = Dense(encoding_dim, activation='relu')(x)
        x = Dense(encoding_dim * 2, activation='relu')(code)
        x = Dropout(dropout)(x)
        x = Dense(encoding_dim * 4, activation='relu')(x)
        x = Dropout(dropout)(x)
        out = Dense(input_dim, activation='linear')(x)
        model = Model(inp, out)
        model.compile(optimizer=Adam(1e-3), loss='mse', metrics=['mae'])
        return model

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        if TF_AVAILABLE:
            try:
                normal_idx = y_train[y_train == 0].index
                Xn = X_train.loc[normal_idx].values
                ae = self._create_autoencoder(Xn.shape[1])
                cb = [
                    EarlyStopping(monitor='loss', patience=8, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='loss', factor=0.5, patience=4, min_lr=1e-6)
                ]
                ae.fit(Xn, Xn, epochs=40, batch_size=64, verbose=0, callbacks=cb)
                self.models['autoencoder_dense'] = ae
            except Exception:
                pass
        self.models['isolation_forest'] = IsolationForest(
            n_estimators=200, contamination=max(0.001, float(y_train.mean())), random_state=self.random_state, n_jobs=-1
        ).fit(X_train.values)
        try:
            self.models['one_class_svm'] = OneClassSVM(kernel='rbf', gamma='scale', nu=min(0.1, max(0.005, float(y_train.mean()) * 2))).fit(
                X_train.loc[y_train == 0].values
            )
        except Exception:
            pass
        try:
            self.models['local_outlier_factor'] = LocalOutlierFactor(n_neighbors=min(20, max(5, len(X_train)//100)), novelty=True)
            self.models['local_outlier_factor'].fit(X_train.loc[y_train == 0].values)
        except Exception:
            pass
        median = np.median(X_train.values, axis=0)
        mad = np.median(np.abs(X_train.values - median), axis=0)
        self.models['statistical'] = {'method': 'modified_z', 'median': median, 'mad': mad, 'threshold': 3.5}

    # ------------------ Prediction helpers ------------------
    def _predict_autoencoder(self, model, X: np.ndarray):
        Xp = model.predict(X, verbose=0)
        errs = np.mean((X - Xp) ** 2, axis=1)
        thr = float(np.percentile(errs, 95))
        pred = (errs > thr).astype(int)
        return pred, errs, thr

    def _predict_statistical(self, params: dict, X: np.ndarray) -> np.ndarray:
        z = 0.6745 * (X - params['median']) / (params['mad'] + 1e-9)
        z = np.abs(z)
        return (np.max(z, axis=1) > params['threshold']).astype(int)

    def predict_all(self, X_test: pd.DataFrame, y_test: Optional[pd.Series] = None) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        X = X_test.values
        if 'autoencoder_dense' in self.models:
            try:
                p, s, t = self._predict_autoencoder(self.models['autoencoder_dense'], X)
                results['autoencoder_dense'] = {'pred': p, 'scores': s, 'threshold': t}
            except Exception:
                pass
        if 'isolation_forest' in self.models:
            m = self.models['isolation_forest']
            p = np.where(m.predict(X) == -1, 1, 0)
            sc = -m.decision_function(X)
            results['isolation_forest'] = {'pred': p, 'scores': sc}
        if 'one_class_svm' in self.models:
            try:
                m = self.models['one_class_svm']
                p = np.where(m.predict(X) == -1, 1, 0)
                sc = -m.decision_function(X)
                results['one_class_svm'] = {'pred': p, 'scores': sc}
            except Exception:
                pass
        if 'local_outlier_factor' in self.models:
            try:
                m = self.models['local_outlier_factor']
                p = np.where(m.predict(X) == -1, 1, 0)
                sc = -m.score_samples(X)
                results['local_outlier_factor'] = {'pred': p, 'scores': sc}
            except Exception:
                pass
        if 'statistical' in self.models:
            p = self._predict_statistical(self.models['statistical'], X)
            results['statistical'] = {'pred': p, 'scores': None}
        if y_test is not None:
            yt = y_test.values
            for name, res in results.items():
                ypred = res['pred']
                metrics = {
                    'accuracy': float(accuracy_score(yt, ypred)),
                    'precision': float(precision_score(yt, ypred, zero_division=0)),
                    'recall': float(recall_score(yt, ypred, zero_division=0)),
                    'f1_score': float(f1_score(yt, ypred, zero_division=0))
                }
                if res['scores'] is not None:
                    try:
                        metrics['auc_roc'] = float(roc_auc_score(yt, res['scores']))
                        metrics['avg_precision'] = float(average_precision_score(yt, res['scores']))
                    except Exception:
                        pass
                self.model_performance[name] = metrics
                res['metrics'] = metrics
        return results

    def predict_ensemble(self, model_results: Dict[str, Any], method: str = 'majority_vote', weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        preds = [v['pred'] for v in model_results.values() if 'pred' in v]
        if not preds:
            return np.zeros(0, dtype=int)
        arr = np.array(preds)
        if method == 'majority_vote':
            return (np.mean(arr, axis=0) > 0.5).astype(int)
        if method == 'weighted' and weights:
            keys = list(model_results.keys())
            w = np.array([weights.get(k, 1.0) for k in keys]).reshape(-1, 1)
            return (np.sum(arr * w, axis=0) / max(w.sum(), 1e-9) > 0.5).astype(int)
        return (np.mean(arr, axis=0) > 0.5).astype(int)

    # ------------------ Persistence ------------------
    def save(self, out_dir: str = 'models') -> None:
        d = Path(out_dir)
        d.mkdir(exist_ok=True)
        for name, m in self.models.items():
            if name.startswith('autoencoder') and TF_AVAILABLE:
                try:
                    m.save(d / f'{name}.h5')
                except Exception:
                    pass
            else:
                try:
                    joblib.dump(m, d / f'{name}.pkl')
                except Exception:
                    pass
        with open(d / 'selected_features.json', 'w') as f:
            json.dump(self.selected_features, f)
        if self.scaler is not None:
            joblib.dump(self.scaler, d / 'scaler.pkl')

    def load(self, in_dir: str = 'models') -> None:
        d = Path(in_dir)
        self.models = {}
        for p in d.glob('*.pkl'):
            name = p.stem
            if name == 'scaler':
                continue
            try:
                self.models[name] = joblib.load(p)
            except Exception:
                pass
        if TF_AVAILABLE:
            for p in d.glob('*.h5'):
                try:
                    self.models[p.stem] = tf.keras.models.load_model(p)
                except Exception:
                    pass
        sf = d / 'selected_features.json'
        if sf.exists():
            try:
                self.selected_features = json.loads(sf.read_text())
            except Exception:
                pass
        sc = d / 'scaler.pkl'
        if sc.exists():
            try:
                self.scaler = joblib.load(sc)
            except Exception:
                pass

    # ------------------ High-level convenience ------------------
    def fit_evaluate(self, data_path: str, test_size: float = 0.2, scaler: str = 'standard', n_features: int = 50) -> Dict[str, Any]:
        df = self.load_all_data(data_path)
        eda = self.quick_eda(df)
        pp = self.preprocess(df, test_size=test_size, scaler=scaler, n_features=n_features)
        self.train_models(pp['X_train'], pp['y_train'])
        results = self.predict_all(pp['X_test'], pp['y_test'])
        return {'eda': eda, 'preprocessing': asdict(pp['info']), 'results': results, 'performance': self.model_performance}


# ------------------ CLI ------------------

def cli_train(args):
    pipe = AICDPipeline()
    out = pipe.fit_evaluate(args.data_path, test_size=args.test_size, scaler=args.scaler, n_features=args.n_features)
    Path('results').mkdir(exist_ok=True)
    with open('results/summary.json', 'w') as f:
        json.dump(out, f, indent=2)
    pipe.save('models')
    print("Training complete. Results -> results/summary.json | Models -> models/")


def cli_predict(args):
    pipe = AICDPipeline()
    pipe.load('models')
    df = pd.read_csv(args.input, sep=args.separator)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = pipe._remove_problematic_columns(df)
    df = pipe._handle_missing(df)
    if pipe.target_column in df.columns:
        df = df.drop(columns=[pipe.target_column])
    X = df[pipe.selected_features] if pipe.selected_features else df.select_dtypes(include=[np.number])
    if pipe.scaler is not None and not X.empty:
        X = pd.DataFrame(pipe.scaler.transform(X), columns=X.columns)
    preds = pipe.predict_all(X)
    ens = pipe.predict_ensemble(preds)
    out_df = pd.DataFrame({'anomaly_prediction': ens})
    out_df.to_csv(args.output, index=False)
    print(f"Predictions saved -> {args.output}")


def main():
    parser = argparse.ArgumentParser(description='AICD Anomaly Detection - Single Pipeline')
    sub = parser.add_subparsers(dest='cmd')

    p_train = sub.add_parser('train')
    p_train.add_argument('--data_path', type=str, default='authenticIndustrialCloudDataDataset/data')
    p_train.add_argument('--test_size', type=float, default=0.2)
    p_train.add_argument('--scaler', type=str, default='standard', choices=['standard', 'robust', 'minmax'])
    p_train.add_argument('--n_features', type=int, default=50)
    p_train.set_defaults(func=cli_train)

    p_pred = sub.add_parser('predict')
    p_pred.add_argument('--input', type=str, required=True)
    p_pred.add_argument('--output', type=str, default='predictions.csv')
    p_pred.add_argument('--separator', type=str, default=';')
    p_pred.set_defaults(func=cli_predict)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

