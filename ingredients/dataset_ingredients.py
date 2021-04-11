from typing import Optional
import pandas as pd
from sacred import Ingredient

dataset_ingredient = Ingredient("dataset")
dataset_ingredient.add_config("config.yaml")


@dataset_ingredient.capture
def get_input(train_data_path: str, test_data_path: Optional[str] = None, target_name: Optional[str] = None):
    """
    Функция загрузки данных.

    Parameters
    ----------
    train_data_path: str
        Путь до обучающей выборки.

    test_data_path: str, optional, default = None
        Путь до тестовой выборки. Опциональный параметр. По умолчанию не
        используется, т.е. тестовая выборка не загружается.

    target_name: str, optional, default = None
        Название целевой переменной.

    Returns
    -------
    train, target: Tuple[pd.DataFrame, pd.Series]
        Кортеж, где первый элемент - матрица признаков, второй - вектор целевой переменной.

    """
    train = pd.read_csv(train_data_path)
    target = train[target_name]

    train = train.drop(target_name, axis=1)
    data_stats(train, target)

    if test_data_path:
        test = pd.read_csv(test_data_path)
        data_stats(test)

        train = train.append(test)
        train = train.reset_index(drop=True)

    return train, target


def data_stats(features, target=None):
    """
    Вывод основных статистик о наборе данных и векторе целевой переменной.

    Parameters
    ----------
    features: pd.DataFrame, shape = [n_samples, n_features]
        Матрица признаков.

    target: pd.Series, optional, default = None
        Вектор целевой переменной.

    """
    print("data.shape = {} rows, {} cols".format(*features.shape))
    print(features.dtypes.value_counts().T)

    if isinstance(target, pd.Series):
        eventrate, events = target.mean(), target.sum()
        print(f"eventrate = {100 * round(eventrate, 4)}%, events = {events}")