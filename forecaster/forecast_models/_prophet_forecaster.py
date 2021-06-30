from forecaster.abstract import AbstractForecaster
from fbprophet import Prophet
import pandas as pd


class ProphetForecaster(AbstractForecaster):

    def __init__(self):
        super().__init__()
        self.model = Prophet()

    def _fit(self, X, y):
        """
        Обучение модели
        :param X: Список дат
        :param y: Список значений
        """
        df = pd.DataFrame()
        df['ds'] = X
        df['y'] = y
        self.model.fit(df)

    def _predict(self, num):
        """
        Предсказание
        :param num: Количество дней, на которые надо предсказать
        :return: Список значений предсказаний
        """
        return self.model.predict(self.model.make_future_dataframe(num))['yhat']