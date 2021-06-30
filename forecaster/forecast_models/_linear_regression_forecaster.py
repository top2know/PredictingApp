from forecaster.abstract import AbstractForecaster
from sklearn.linear_model import LinearRegression
import numpy as np


class LinearRegressionForecaster(AbstractForecaster):

    def __init__(self):
        super().__init__()
        self.model = LinearRegression()
        self.min_num = None

    def _fit(self, X, y):
        """
        Обучение модели
        :param X: Список дат
        :param y: Список значений
        """
        X = list(map(lambda x: [x.timestamp()], X))
        self.min_num = np.min(X)
        self.model.fit(X, y)

    def _predict(self, num):
        """
        Предсказание
        :param num: Количество дней, на которые надо предсказать
        :return: Список значений предсказаний
        """
        upper_bound = self.max_date.timestamp() + num * 86400
        return self.model.predict(list(map(lambda x: [x], np.arange(self.min_num, upper_bound + 1, 86400))))
