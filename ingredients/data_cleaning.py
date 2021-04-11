import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sacred import Ingredient

cleaner_ingredient = Ingredient("data_cleaning")
cleaner_ingredient.add_config("config.yaml")


class ApplicationCleaning(BaseEstimator, TransformerMixin):
    """
    Очистка данных из источника application_train / application_test.

    Parameters
    ----------
    fill_missing: bool, optional, default = False
        Флаг заполнения пропусков. Опциональный параметр, по умолчанию, не используется.

    fill_value: float, optional, default = 0
        Значение для заполнения пропусков.

    copy: bool, optional, default = True
        Если True, то для преобразования используется копия данных, иначе исходный набор
        данных. Опциональный параметр, по умолчанию, используется копия данных.

    """
    def __init__(self, fill_missing: bool = False, fill_value: float = 0, copy: bool = True) -> None:
        self.fill_missing = fill_missing
        self.fill_value = fill_value
        self.copy = copy

    def _copy(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.copy() if self.copy else X

    def fit(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Очистка данных.

        Parameters
        ----------
        X: pandas.DataFrame, shape = [n_samples, n_features]
            Матрица признаков.

        Returns
        -------
        X_transformed: pandas.DataFrame, shape = [n_samples, n_features]
            Преобразованная матрица признаков.

        """
        X = self._copy(X)
        X["CODE_GENDER"] = X["CODE_GENDER"].replace("XNA", np.nan)
        X["DAYS_EMPLOYED"] = X["DAYS_EMPLOYED"].replace(365243, np.nan)
        X["DAYS_LAST_PHONE_CHANGE"] = X["DAYS_LAST_PHONE_CHANGE"].replace(0, np.nan)
        X["NAME_FAMILY_STATUS"] = X["NAME_FAMILY_STATUS"].replace("Unknown", np.nan)
        X["ORGANIZATION_TYPE"] = X["ORGANIZATION_TYPE"].replace("XNA", np.nan)

        if self.fill_missing:
            X = X.fillna(self.fill_value)

        return X


@cleaner_ingredient.capture
def apply_cleaners(cleaner, fill_missing, fill_value, X, y=None):
    cleaner_ = cleaner(fill_missing, fill_value)
    return cleaner_.fit_transform(X, y)
