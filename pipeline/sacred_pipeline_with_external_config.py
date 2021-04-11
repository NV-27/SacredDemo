import sys
import time
from sacred import Experiment
from sacred.observers import FileStorageObserver
sys.path.append("../")

from src.utils import load_data
from src.models import CatBoostClassifierCV
from src.validation import CrossValidationSplitter

ex = Experiment(name="home-credit-risk-default-base-sacred-pipeline")
ex.observers.append(FileStorageObserver("../runs"))
ex.add_config("../configs/config.yaml")

@ex.capture
def get_cv(cv_strategy, n_folds, shuffle, cv_seed):
    return CrossValidationSplitter(cv_strategy, n_folds, shuffle, cv_seed)

@ex.capture
def get_estimator(model_type, model_params, used_features, cv):
    if model_type == "catboost":
        return CatBoostClassifierCV(cv, model_params, used_features)
    elif model_type == "xgboost":
        return XGBoostClassifierCV(cv, model_params, used_features)
    elif model_type == "lightgbm":
        return LightGBMClassifierCV(cv, model_params, used_features)
    else:
        raise ValueError("This pipeline only for GBDT-models.")

@ex.automain
def main(_run, filename, target_name):
    print(f"Pipeline started at {time.ctime()}")
    data, target = load_data(filename, target_name, n_rows=20000)
    used_features = data.dtypes[data.dtypes == "float"].index.tolist()

    cv = get_cv()
    model = get_estimator(cv=cv, used_features=used_features)
    model.fit(data, target)

    _run.log_scalar("CV-ROC-AUC", model.cv_score)
    _run.log_scalar("CV-evals_result_", model.evals_result_)
    _run.log_scalar("CV-best_iteration", model.best_iteration)