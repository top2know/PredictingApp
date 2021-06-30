from forecaster.abstract import AbstractForecaster
from statsmodels.tsa.arima_model import ARIMA
import numpy as np


class ARMAForecaster(AbstractForecaster):

    def __init__(self):
        super().__init__()
        self.model = None
        self.train_len = None

    def _fit(self, X, y):
        """
        Обучение модели
        :param X: Список дат
        :param y: Список значений
        """
        self.train_len = len(y)
        self.model = ARIMA(y, order=(7, 0, 2)).fit()

    def _predict(self, num):
        """
        Предсказание
        :param num: Количество дней, на которые надо предсказать
        :return: Список значений предсказаний
        """
        return list(np.concatenate([[None] * self.train_len, self.model.forecast(num)[0]]))