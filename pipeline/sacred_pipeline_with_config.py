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


@ex.config
def config():
    filename = "/Users/nv27/Documents/other_ml/data/home-credit-default-risk/application_train.csv"
    target_name = "TARGET"
    n_rows = 20000
    model_params = {
        "n_estimators": 100,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "task_type": "CPU",
        "max_depth": 6,
        "max_bin": 20,
        "verbose": 10,
        "l2_leaf_reg": 100,
        "early_stopping_rounds": 50,
        "random_seed": 42
    }
    cv = CrossValidationSplitter(
        cv_strategy="kfold", n_folds=3, cv_seed=27
    )


@ex.automain
def main(filename, target_name, n_rows, model_params, cv):
    print(f"Pipeline started at {time.ctime()}")
    data, target = load_data(filename, target_name, n_rows=n_rows)
    used_features = data.dtypes[data.dtypes == "float"].index.tolist()
    print(data[used_features].head(n=5))
    print(target.head(n=5))

    model = CatBoostClassifierCV(cv, model_params, used_features)
    model.fit(data, target)