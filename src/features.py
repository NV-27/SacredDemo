from sklearn.base import BaseEstimator, TransformerMixin


class ApplicationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, copy: bool = True):
        self.copy = copy

    def _copy(self, X):
        return X.copy(deep=True) if self.copy else X

    def fit(self, X, y):
        return self

    def transform(self, X, y):
        x_transformed = self._copy(X)
        features = x_transformed.dtypes[x_transformed.dtypes == "float"]
        features = features.index.tolist()

        return x_transformed[features]
