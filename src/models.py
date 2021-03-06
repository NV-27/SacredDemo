import time
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import pandas as pd
import catboost as cb
from sklearn.utils.validation import check_is_fitted


class BaseClassifier(ABC):
    """
    Интерфейс базового классификатора.

    """
    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @property
    def best_iteration(self):
        pass

    @property
    def cv_score(self):
        pass

    @property
    def best_iterations_(self):
        pass

    @property
    def evals_result_(self):
        pass


class CatBoostClassifierCV(BaseClassifier):
    """
    Модель CatBoost для проведения кросс-валидации.

    Parameters
    ----------
    cv_splitter: src.validation.CrossValidationSplitter
        Стратегия кросс-валидации.

    params: dict
        Словарь гиперпараметров модели.

    used_features: List[str]
        Список используемых для обучения признаков.

    categorical_features: List[str], optional, default = None
        Список категориальных признаков. Опциональный параметр. По умолчанию, не используется.

    fit_params: dict
        Словарь параметров обучения.

    Attributes
    ----------
    estimators: list
        Список экземпляров обученных на фолдах моделей.

    """
    def __init__(self,
                 cv_splitter,
                 params: dict,
                 used_features: List[str],
                 categorical_features: Optional[List[str]] = None,
                 fit_params: Optional[dict] = None):

        self.cv_splitter = cv_splitter
        self.params = params
        self.used_features = used_features
        self.categorical_features = categorical_features
        self.fit_params = fit_params
        self.estimators = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Кросс-валидация для модели CatBoost.

        Parameters
        ----------
        X: pandas.DataFrame, shape = [n_samples, n_features]
            Матрица признаков для обучения.

        y: pandas.Series, shape  [n_samples, ]
            Вектор целевой переменной.

        Returns
        -------
        self

        """
        self.estimators = []
        cv = self.cv_splitter.transform(X, y)
        print(f"{time.ctime()}, start CatBoost-CV, with {len(X)} rows, {X.shape[1]} cols.")

        for fold, (train, valid) in enumerate(cv):
            print(f"Fold: {fold + 1}")
            x_train, x_valid = X.loc[train, self.used_features], X.loc[valid, self.used_features]
            y_train, y_valid = y[train], y[valid]

            self.estimators.append(
                cb.CatBoostClassifier(**self.params).fit(
                    x_train, y_train, self.categorical_features, eval_set=[(x_train, y_train), (x_valid, y_valid)]
                )
            )
        return self

    def predict(self, X: pd.DataFrame) -> np.array:
        """
        Применение обученных моделей к новому набору данных.

        Parameters
        ----------
        X: pandas.DataFrame, shape = [n_samples, n_features]
            Матрица признаков для применения модели.

        Returns
        -------
        y_pred: np.array, shape = [n_samples, ]
            Вектор прогнозов.

        """
        y_pred = []
        check_is_fitted(self, "estimators")
        for estimator in self.estimators:
            y_pred.append(estimator.predict_proba(X))
        return np.mean(y_pred, axis=0)

    @property
    def best_iteration(self):
        """
        Вычисление оптимального количества итераций обучения.
        """
        return np.mean(self.best_iterations_)

    @property
    def cv_score(self):
        """
        Вычисление метрики на кросс-валидации.
        """
        return np.mean(self.evals_result_)

    @property
    def evals_result_(self):
        """
        Вычисление метрики на каждом фолде обучения.
        """
        evals_result = []
        metric_name = self.params["eval_metric"]
        for estimator in self.estimators:
            evals_result.append(
                estimator.best_score_["validation_1"][metric_name]
            )
        return np.array(evals_result)

    @property
    def best_iterations_(self):
        """
        Вычисление оптимального числа итераций на каждом фолде обучения.
        """
        best_iterations = []
        for estimator in self.estimators:
            best_iterations.append(
                estimator.best_iteration_
            )
        return np.array(best_iterations)
