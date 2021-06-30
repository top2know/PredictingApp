from datetime import datetime
import numpy as np


class AbstractForecaster:

    def __init__(self):
        self.model = None
        self.is_fit = False
        self.max_date = None

    def _fit(self, X, y):
        """
        Обучение модели - метод для переопределения
        :param X: Список дат
        :param y: Список значений
        """
        self.model.fit(X, y)

    def fit(self, X, y):
        """
        Обучение модели
        :param X: Список дат
        :param y: Список значений
        """
        self._fit(X, y)
        self.max_date = np.max(X)
        self.is_fit = True
        pass

    def _predict(self, num):
        """
        Предсказание
        :param num: Количество дней, на которые надо предсказать
        :return: Список значений предсказаний
        """
        pass

    def predict(self, date_to=None):
        """
        Предсказание
        :param date_to: Дата, до которой надо предсказать
        :return: Список значений предсказаний
        """
        if not self.is_fit:
            raise ValueError('Модель не обучена!')
        if date_to is None:
            num = 30
        else:
            num = (date_to - self.max_date).days

        return self._predict(num)

    def fit_predict(self, X, y, date_to):
        """
        Единый метод для обучения и предсказания
        :param X: Список дат
        :param y: Список значений
        :param date_to: Дата, до которой необходимы предсказания
        :return: Список значений предсказаний
        """
        try:
            self.fit(X, y)
            return self.predict(date_to)
        except ValueError as e:
            raise ValueError(e.args[0])
        except Exception as e:
            raise ValueError('Произошла неизвестная ошибка: ' + e.args[0])
