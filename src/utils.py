from typing import Optional, List
import numpy as np
import pandas as pd


def load_data(filename: str,
              target_name: Optional[str] = None,
              used_features: Optional[List[str]] = None,
              n_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Загрузка данных с диска.

    Parameters
    ----------
    filename: str
        Путь до загружаемого файла.

    target_name: str, optional, default = None
        Название признака с целевой переменной. Опциональный параметр.
        По умолчанию не используется. Если значение задано, то функция
        возвращает матрицу признаков и вектор целевой переменной, иначе -
        - только матрицу признаков.

    used_features: List[str], optional, default = None
        Список используемых признаков. Опциональный параметр.
        По умолчанию не используется. Если значение задано, то
        используются только заданные признаки, иначе - используются все
        признаки датасета.

    n_rows: int, optional, default = None
        Количество загружаемых строк. Опциональный параметр.
        По умолчанию не используется, и загружается весь набор данных.

    Returns
    -------
    data or data, target: pd.DataFrame or Tuple[pd.DataFrame, pd.Series]
        Матрица признаков или матрица признаков и вектор целевой переменной.

    """
    data = pd.read_csv(filename, nrows=n_rows)

    if not used_features:
        used_features = data.columns.to_list()
        used_features.remove(target_name)

    if target_name:
        target = data[target_name]
        data = data.drop(target_name, axis=1)

        n_features, obs, events, eventrate = len(used_features), target.shape[0], target.sum(), target.mean()
        print(f"obs = {obs}, features = {n_features}, events = {events}, eventrate = {100 * np.round(eventrate, 4)}%")
        return data[used_features], target

    return data[used_features]
