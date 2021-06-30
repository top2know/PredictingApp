from forecaster.abstract import AbstractForecaster
import numpy as np


class LogarithmicModel:
    def __init__(self):
        self.a = None
        self.b = None

    def fit(self, X, y):
        print(X, y)
        self.a, self.b = np.polyfit(np.log(X), y, 1)

    def predict(self, X):
        print(X)
        return self.a * (np.log(X)) + self.b


class LogarithmicRegressionForecaster(AbstractForecaster):

    def __init__(self):
        super().__init__()
        self.model = LogarithmicModel()
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
        return self.model.predict(list(map(lambda x: x, np.arange(1, upper_bound, 1))))
