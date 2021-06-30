from forecaster.abstract import AbstractForecaster
import numpy as np


class SquaredModel:
    def __init__(self):
        self.a = None
        self.b = None
        self.c = None

    def fit(self, X, y):
        self.a, self.b, self.c = np.polyfit(X, y, 2)

    def predict(self, X):
        return list(self.a * np.array(X)**2 + self.b * np.array(X) + self.c)


class SquaredRegressionForecaster(AbstractForecaster):

    def __init__(self):
        super().__init__()
        self.model = SquaredModel()
        self.min_num = None

    def _fit(self, X, y):
        """
        Обучение модели
        :param X: Список дат
        :param y: Список значений
        """
        X = list(map(lambda x: x.timestamp(), X))
        self.min_num = np.min(X)
        self.model.fit((X - self.min_num) / 86_400 + 2, y)

    def _predict(self, num):
        """
        Предсказание
        :param num: Количество дней, на которые надо предсказать
        :return: Список значений предсказаний
        """
        upper_bound = (self.max_date.timestamp() - self.min_num) / 86_400 + num + 2
        return self.model.predict(list(map(lambda x: x, np.arange(2, upper_bound + 1, 1))))
