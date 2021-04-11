import time
from sacred import Experiment
from sacred.observers import FileStorageObserver
from ingredients_func.dataset_ingredients import dataset_ingredient, get_input
from ingredients_func.data_cleaning import cleaner_ingredient, ApplicationCleaning, apply_cleaners

from src.validation import CrossValidationSplitter
from src.models import CatBoostClassifierCV


ex = Experiment(name="kaggle-home-credit-competition-pipeline", ingredients=[dataset_ingredient, cleaner_ingredient])
ex.observers.append(FileStorageObserver("runs"))
ex.add_config("config.yaml")


@ex.capture
def get_cv(cv_strategy, n_folds, shuffle, cv_seed):
    return CrossValidationSplitter(cv_strategy, n_folds, shuffle, cv_seed)

@ex.automain
def main(_run):
    _run.add_artifact("scrd.py")
    _run.add_artifact("ingredients_func/dataset_ingredients.py")
    _run.add_artifact("ingredients_func/data_cleaning.py")
    _run.add_artifact("model.pkl")
    print(f"{time.ctime()}, pipeline start.")

    data, target = get_input()
    splitter = get_cv()

    catboost_params = {
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
    used_features = data.dtypes[data.dtypes=="float"].index.tolist()
    model = CatBoostClassifierCV(splitter, catboost_params, used_features)
    model.fit(data, target)

    _run.log_scalar("CV-ROC-AUC", model.cv_score)
    _run.log_scalar("CV-best_iteration", model.best_iteration)