from forecaster.forecast_models import *


class PredictorsFactory:

    def __init__(self):
        pass

    @staticmethod
    def get_all_methods():
        return ['linreg', 'logreg', 'sqreg', 'prophet', 'arima', 'arma']

    @staticmethod
    def get_model(name):
        if name == 'prophet':
            return ProphetForecaster()
        if name == 'linreg':
            return LinearRegressionForecaster()
        if name == 'logreg':
            return LogarithmicRegressionForecaster()
        if name == 'sqreg':
            return SquaredRegressionForecaster()
        if name == 'arima':
            return ARIMAForecaster()
        if name == 'arma':
            return ARMAForecaster()
