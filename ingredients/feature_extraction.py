import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ApplicationFeatures(BaseEstimator, TransformerMixin):
    """
    Создание новых признаков на основе application_train / application_test.


    """
    def __init__(self, categorical_features, numerical_features, copy: bool = True):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.copy = copy

    def _copy(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.copy() if self.copy else X

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(selfs, X: pd.DataFrame, y=None):
        X = self._copy(X)
        X["ANNUITY_INCOME_PERCENTAGE"] = X["AMT_ANNUITY"] / X["AMT_INCOME_TOTAL"]

        X["CREDIT_TO_ANNUITY_RATIO"] = X["AMT_CREDIT"] / X["AMT_ANNUITY"]
        X["CREDIT_TO_GOODS_RATIO"] = X["AMT_CREDIT"] / X["AMT_GOODS_PRICE"]
        X["CREDIT_TO_INCOME_RATIO"] = X["AMT_CREDIT"] / X["AMT_INCOME_TOTAL"]
        X["CREDIT_PER_PERSON"] = X["AMT_CREDIT"] / X["CNT_FAM_MEMBERS"]
        X["CREDIT_PER_CHILD"] = X["AMT_CREDIT"] / (1 + X["CNT_CHILDREN"])

        X["INCOME_CREDIT_PERCENTAGE"] = X["AMT_INCOME_TOTAL"] / X["AMT_CREDIT"]
        X["INCOME_PER_CHILD"] = X["AMT_INCOME_TOTAL"] / (X["CNT_CHILDREN"] + 1)
        X["INCOME_PER_PERSON"] = X["AMT_INCOME_TOTAL"] / X["CNT_FAM_MEMBERS"]

        X["PAYMENT_RATIO"] = X["AMT_ANNUITY"] / X["AMT_CREDIT"]
        X["CHILDREN_RATIO"] = X["CNT_CHILDREN"] / X["CNT_FAM_MEMBERS"]
        X["EXTERNAL_SOURCES_WEIGHTED"] = np.nansum(
            np.array([1.9, 2.1, 2.6]) * X[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]], axis=1
        )
        X["CNT_NON_CHILD"] = X["CNT_FAM_MEMBERS"] - X["CNT_CHILDREN"]

        return X
