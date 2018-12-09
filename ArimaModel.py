import warnings
from math import sqrt
from statistics import mean

import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error


class ArimaModel:
    """Custom ArimaModel class"""

    def __init__(self, series_train_data, validation_criteria, data_frequency=0):
        """

        Parameters
        ----------
        series_train_data: list of values of time series
        validation_criteria: 'aic' for Akaike Information Criteria, 'cross' for cross validation with rmse
        data_frequency: interval of data (12 for monthly series, 4 for quaternal); if value bigger than 1 is provided,
                        then a seasonal model will be used; values lower than 2 are ignored

        """
        self.series_train_data = series_train_data
        self.validation_criteria = validation_criteria
        self.data_frequency = data_frequency

        self.series_size = len(self.series_train_data)

        # maximal values of standard arima model parameters
        self.max_p = 5
        self.max_d = 5
        self.max_q = 5

        # maximal values of seasonal arima model parameters
        self.max_P = 2
        self.max_D = 1
        self.max_Q = 2

        # significance level for KPSS test
        self.significance_level = 0.05

        if self.validation_criteria == 'aic':
            self.validation_function = self.__aic
        else:
            self.validation_function = self.__time_series_cross_validation
            self.max_nr_of_folds = 6

        # do not use seasonal component if series is of yearly frequency
        if data_frequency < 2:
            self.data_frequency = 0

        # parameters of arima model
        self.p, self.d, self.q, self.P, self.D, self.Q = self.__find_parameters()

        # fit a model with found parameters
        self.model = self.__fit_model()

    def __find_parameters(self, train_data=None):
        """Determine values of arima model parameters according to Hyndman-Khandakar procedure"""
        
        if train_data is None:
            train_data = self.series_train_data

        train_data_size = len(train_data)

        # seasonal models
        if self.data_frequency > 0:
            possible_P_values = range(self.max_P + 1)
            possible_D_values = range(self.max_D + 1)
            possible_Q_values = range(self.max_Q + 1)
            # initial sets of p, q, P, Q values in Hyndman-Khandakar procedure for seasonal models
            start_parameters_sets = [(0, 0, 0, 0), (1, 0, 1, 0), (0, 1, 0, 1), (2, 2, 1, 1)]
        # nonseasonal models
        else:
            possible_P_values = [0]
            possible_D_values = [0]
            possible_Q_values = [0]
            # initial sets of p, q, P, Q values in Hyndman-Khandakar procedure for nonseasonal models
            start_parameters_sets = [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (2, 2, 0, 0)]

        possible_p_values = range(self.max_p + 1)
        possible_q_values = range(self.max_q + 1)

        best_parameters_list = []

        # since there is no extended Canova-Hansen test implemented in Python we will simply follow Hyndman-Khandakar
        # procedure for every possible D value and choose the best model according to chosen validation_criteria
        for D in possible_D_values:

            # set d parameter according to kpss test
            d = self.__find_d_parameter(D)

            validation_values_dict = {}
            # get only these initial sets of parameters for which there is sufficient amount of train data to fit model
            parameters_sets = [(p, q, P, Q) for p, q, P, Q in start_parameters_sets
                               if self.__sufficient_train_data_size(train_data_size, p, d, q, P, D, Q)]
            for p, q, P, Q in parameters_sets:
                if (p, q, P, Q) not in validation_values_dict:
                    validation_values_dict[(p, q, P, Q)] = \
                        self.validation_function(train_data, (p, d, q), (P, D, Q, self.data_frequency))

            best_p, best_q, best_P, best_Q = min(validation_values_dict, key=validation_values_dict.get)
            best_value = validation_values_dict[(best_p, best_q, best_P, best_Q)]
            parameters_sets = self.__next_sets_of_parameters(best_p, best_q, best_P, best_Q)

            # in case of nonseasonal models sets with P and Q other than 0 are dropped
            parameters_sets = [(p, q, P, Q) for p, q, P, Q in parameters_sets
                               if p in possible_p_values and q in possible_q_values
                               and P in possible_P_values and Q in possible_Q_values
                               and self.__sufficient_train_data_size(train_data_size, p, d, q, P, D, Q)]

            # remove sets of parameters for which we have insufficient amount of train data to fit the model reasonably

            while parameters_sets:
                for p, q, P, Q in parameters_sets:
                    if (p, q, P, Q) not in validation_values_dict:
                        try:
                            validation_values_dict[(p, q, P, Q)] = \
                                self.validation_function(train_data, (p, d, q), (P, D, Q, self.data_frequency))
                        except Exception as exception:
                            print('Evaluating model with parameters: {} {} failed due to exception: {}. Skipping this '
                                  'set of parameters.'.format((p, d, q), (P, D, Q, self.data_frequency), exception))
                            # if evaluating a model with given parameters was unsuccessful then add them to dict
                            # with best_value so far -> they will not be used as smaller value is searched
                            validation_values_dict[(p, q, P, Q)] = best_value

                min_value = min(validation_values_dict.values())

                if min_value < best_value:
                    best_p, best_q, best_P, best_Q = min(validation_values_dict, key=validation_values_dict.get)

                    best_value = min_value
                    parameters_sets = self.__next_sets_of_parameters(best_p, best_q, best_P, best_Q)

                    parameters_sets = [(p, q, P, Q) for p, q, P, Q in parameters_sets
                                       if p in possible_p_values and q in possible_q_values
                                       and P in possible_P_values and Q in possible_Q_values
                                       and self.__sufficient_train_data_size(train_data_size, p, d, q, P, D, Q)]

                else:
                    break

            best_parameters_list.append((best_value, best_p, d, best_q, best_P, D, best_Q))

        _, p, d, q, P, D, Q = min(best_parameters_list, key=lambda x: x[0])

        return p, d, q, P, D, Q

    def __find_d_parameter(self, D):
        """Find number of differences d according to KPSS statistical test"""

        # set the starting number of differences
        d = 0
        differentiated_data = self.series_train_data.copy()

        # if model has a seasonal component then apply seasonal differences first
        for _ in range(D):
            differentiated_data = [differentiated_data[i] - differentiated_data[i - self.data_frequency]
                                   for i in range(self.data_frequency, len(differentiated_data))]

        while d < self.max_d:
            # suppress warning about p-value outside the table of critical values
            # they mean that either p-value is greater than 0.1 or smaller than 0.01 which is irrelevant here
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, kpss_p_value, _, _ = sm.tsa.stattools.kpss(differentiated_data)
            if kpss_p_value > self.significance_level:
                break
            differentiated_data = [differentiated_data[i] - differentiated_data[i - 1]
                                   for i in range(1, len(differentiated_data))]
            d += 1

        return d

    def __next_sets_of_parameters(self, p, q, P, Q):
        """
        Return new variations of arima parameters according to Hyndman-Khandakar procedure:
            where one of p, q, P and Q is allowed to vary by ±1 from the current model or
            where p and q both vary by ±1 from the current model or
            where P and Q both vary by ±1 from the current model
        """

        return [(p + 1, q, P, Q),
                (p - 1, q, P, Q),
                (p, q + 1, P, Q),
                (p, q - 1, P, Q),
                (p, q, P + 1, Q),
                (p, q, P - 1, Q),
                (p, q, P, Q + 1),
                (p, q, P, Q - 1),
                (p + 1, q + 1, P, Q),
                (p - 1, q - 1, P, Q),
                (p, q, P + 1, Q + 1),
                (p, q, P - 1, Q - 1)]

    def __sufficient_train_data_size(self, train_data_size, p, d, q, P, D, Q):
        """Check if train data size is sufficient to fit a model with given set of parameters"""
        # minimal size of series train data needed to fit a model with given parameters
        minimal_data_size = d + D * self.data_frequency + \
                            max(3 * q + 1, 3 * Q * self.data_frequency + 1, p, P * self.data_frequency) + 1

        if self.validation_criteria == 'aic':
            return minimal_data_size <= train_data_size
        # if cross validation is used for model validation check if we have at least twice as much data
        else:
            return minimal_data_size <= int(train_data_size / 2)

    def __aic(self, train_data, parameters, seasonal_parameters):
        """Fit a model with its train data and provided arguments and return Akaike Information Criteria"""
        fitted_model = self.__fit_model(train_data, parameters, seasonal_parameters)
        return fitted_model.aic

    def __time_series_cross_validation(self, train_data, parameters, seasonal_parameters):
        """Perform cross validation of time series and return mean of rmse of all models created for validation"""
        fold_nr = 0
        rmse_per_fold = []

        train_data_size = len(train_data)

        # set window_size so that max_nr_of_folds validations can be done
        # and model is fitted on at least half of train data
        window_size = int(train_data_size / 2 / self.max_nr_of_folds) or 1
        window_slide = window_size

        while fold_nr < self.max_nr_of_folds:
            index = train_data_size - window_size - window_slide * fold_nr
            # at least as much fit data as window_size
            if index < window_size:
                break
            fit_data = train_data[:index]
            validation_data = train_data[index:index + window_size]
            fitted_model = self.__fit_model(fit_data, parameters, seasonal_parameters)
            validation_forecast = fitted_model.forecast(window_size).tolist()
            rmse_per_fold.append(sqrt(mean_squared_error(validation_data, validation_forecast)))
            fold_nr += 1
        return mean(rmse_per_fold)

    def __fit_model(self, fit_data=None, parameters=None, seasonal_parameters=None):
        """Fit arima model with given data and parameters"""
        
        if fit_data is None:
            fit_data=self.series_train_data
            
        if parameters is None:
            parameters=(self.p, self.d, self.q)
            
        if seasonal_parameters is None:
            seasonal_parameters=(self.P, self.D, self.Q, self.data_frequency)
            
        # suppress FutureWarning about wrong input data format
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)

        # we use SARIMAX to avoid errors with non stationary series when fitting standard ARIMA model
            arima_model = sm.tsa.statespace.SARIMAX(fit_data,
                                                    order=parameters,
                                                    seasonal_order=seasonal_parameters,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)
            fitted_model = arima_model.fit(maxiter=200, method='powell', disp=False)

        return fitted_model

    def make_prediction(self, series_test_data, forecast_type):
        """Make prediction on train data and forecast of test data using either full or one-step method"""

        # prediction of train data
        prediction = self.model.predict()
        test_data_size = len(series_test_data)

        if forecast_type == 'full':
            # forecast of all test data
            forecast = self.model.forecast(test_data_size)
        elif forecast_type == 'one_step':
            refit_train_data = self.series_train_data.copy()
            # forecast of first value from test data
            forecast = self.model.forecast(1)
            # add expected value to train data
            refit_train_data.append(series_test_data[0])

            for i in range(1, test_data_size):
                # reset arima parameters according to new train data
                p, d, q, P, D, Q = self.__find_parameters(refit_train_data)
                # fit new model on increased train_data
                refitted_model = self.__fit_model(refit_train_data, (p, d, q), (P, D, Q, self.data_frequency))
                # forecast single next value
                forecast = np.append(forecast, refitted_model.forecast(1))
                # append real value for next time step forecast
                refit_train_data.append(series_test_data[i])
        return prediction.tolist(), forecast.tolist()

    def get_parameters(self):
        """Return list of model parameters"""
        return [self.p, self.d, self.q, self.P, self.D, self.Q]
