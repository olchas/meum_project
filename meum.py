import argparse
import os
import time
from math import sqrt

import numpy as np
import pandas as pd
from functools import reduce
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

from ArimaModel import ArimaModel


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, choices=['linear', 'catboost'], default='linear',
                        help="type of time series model")
    parser.add_argument("-i", "--input_file", type=str, default='data/M3C.xls',
                        help="path to M3C file with input data")
    parser.add_argument("-c", "--categories", type=str, nargs='+',
                        choices=['MICRO', 'INDUSTRY', 'MACRO', 'FINANCE', 'DEMOGRAPHIC', 'OTHER'], default=['MICRO'],
                        help="list of categories to use")
    parser.add_argument("-f", "--frequency", type=str, choices=['Year', 'Quart', 'Month', 'Other'], default='Month',
                        help="frequency of series")
    parser.add_argument("-t", "--forecast_type", type=str, choices=['full', 'one_step'], default='full',
                        help="forecast type:"
                             "full to predict all time steps at once (using prediction to forecast further time steps)"
                             "one_step to predict one step at a time (using only true data for every prediction)")
    parser.add_argument("-v", "--validation_criteria", type=str, choices=['aic', 'cross'], default='aic',
                        help="linear model only: whether to use akaike information criteria or cross validation")
    parser.add_argument("-o", "--output_dir", type=str, default='results',
                        help="path to directory to store output files")
    parser.add_argument("--seasonal_arima_model", action='store_true', default=False,
                        help="linear model only: true if you want to use seasonal arima model, "
                             "Quart and Month intervals only")

    return parser.parse_args()


def load_data(input_file, categories, frequency):

    m3c_file = pd.ExcelFile(input_file)

    sheet_name = 'M3' + frequency

    m3c_month_df = m3c_file.parse(sheet_name)

    # strip unnecessary spaces from 'Category' column
    m3c_month_df['Category'] = m3c_month_df['Category'].apply(lambda x: x.strip())

    return m3c_month_df[m3c_month_df['Category'].isin(categories)]


def next_series_generator(data_df, test_data_size):

    scaler = preprocessing.MinMaxScaler()

    for _, row in data_df.iterrows():
        # series_size incremented for the purpose of range
        series_size = row['N'] + 1
        # conversion to type float64 to silence the warning when fitting scaler
        train_data = row.loc[range(1, series_size - test_data_size)].values.astype(np.float64)
        test_data = row.loc[range(series_size - test_data_size, series_size)].values.astype(np.float64)

        # normalise data to range [0, 1]
        scaler.fit(train_data.reshape(-1, 1))
        train_data = scaler.transform(train_data.reshape(1, -1)).tolist()[0]
        test_data = scaler.transform(test_data.reshape(1, -1)).tolist()[0]

        yield train_data, test_data, scaler


if __name__ == '__main__':

    args = parse_arguments()

    data_df = load_data(args.input_file, args.categories, args.frequency)

    if args.frequency == 'Year':
        test_data_size = 6
        data_frequency = 1
    elif args.frequency == 'Quart':
        test_data_size = 8
        data_frequency = 4
    elif args.frequency == 'Month':
        test_data_size = 18
        data_frequency = 12
    else:
        test_data_size = 8
        data_frequency = 0

    if not args.seasonal_arima_model:
        data_frequency = 0

    predicted_data_matrix = []
    expected_data_matrix = []

    normalized_forecast_data_matrix = []
    normalized_test_data_matrix = []

    # list of dicts of parameters of each trained model
    parameters_dicts = []

    rmse_prediction = []
    rmse_forecast = []
    rmse_total = []

    training_time = 0
    testing_time = 0

    for train_data, test_data, scaler in next_series_generator(data_df, test_data_size):

        if args.model == 'linear':

            training_start = time.time()
            model = ArimaModel(train_data, args.validation_criteria, data_frequency)
            training_end = time.time()

            prediction, forecast = model.make_prediction(test_data, args.forecast_type)
            testing_end = time.time()

            training_time += training_end - training_start
            testing_time += testing_end - training_end
            parameters_dicts.append(model.get_parameters())

        elif args.model == 'catboost':
            pass

        # rmse per series
        rmse_prediction.append(sqrt(mean_squared_error(train_data, prediction)))
        rmse_forecast.append(sqrt(mean_squared_error(test_data, forecast)))
        rmse_total.append(sqrt(mean_squared_error(train_data + test_data, prediction + forecast)))

        normalized_forecast_data_matrix.append(forecast)
        normalized_test_data_matrix.append(test_data)

        # denormalization of data
        predicted_data = scaler.inverse_transform([prediction + forecast]).tolist()[0]
        expected_data = scaler.inverse_transform([train_data + test_data]).tolist()[0]

        predicted_data_matrix.append(predicted_data)
        expected_data_matrix.append(expected_data)

    # names of data columns in form of integers
    data_columns = list(range(1, max(len(x) for x in predicted_data_matrix) + 1))
    series_index = list(range(len(predicted_data_matrix)))
    series_length = [len(x) for x in predicted_data_matrix]

    predicted_data_df = pd.DataFrame(predicted_data_matrix, columns=data_columns)
    predicted_data_df['series'] = series_index
    predicted_data_df['N'] = series_length
    predicted_data_df['NF'] = test_data_size

    # take parameters names from first dict
    parameters_names = list(parameters_dicts[0].keys())
    for parameter in parameters_names:
        predicted_data_df[parameter] = [x[parameter] for x in parameters_dicts]

    predicted_data_df['rmse_prediction'] = rmse_prediction
    predicted_data_df['rmse_forecast'] = rmse_forecast
    predicted_data_df['rmse_total'] = rmse_total

    # change the order of columns
    output_columns = ['series', 'N', 'NF'] + parameters_names + ['rmse_prediction', 'rmse_forecast', 'rmse_total'] \
                     + data_columns
    predicted_data_df = predicted_data_df[output_columns]

    expected_data_df = pd.DataFrame(expected_data_matrix, columns=data_columns)
    expected_data_df['series'] = series_index
    expected_data_df['N'] = series_length
    expected_data_df['NF'] = test_data_size

    output_columns = ['series', 'N', 'NF'] + data_columns
    expected_data_df = expected_data_df[output_columns]

    # calculate error separately for each month of forecast
    rmse_per_month = []
    for i in range(test_data_size):
        month_forecast = [x[i] for x in normalized_forecast_data_matrix]
        month_expected = [x[i] for x in normalized_test_data_matrix]
        rmse_per_month.append(sqrt(mean_squared_error(month_expected, month_forecast)))

    # calculate total error of all forecasts
    all_forecast = reduce(lambda x, y: x + y, normalized_forecast_data_matrix)
    all_test = reduce(lambda x, y: x + y, normalized_test_data_matrix)

    total_rmse = sqrt(mean_squared_error(all_test, all_forecast))

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    normalized_forecast_data_df = pd.DataFrame(normalized_forecast_data_matrix)
    normalized_test_dat_df = pd.DataFrame(normalized_test_data_matrix)

    normalized_forecast_data_df.to_csv(os.path.join(args.output_dir, 'normalized_forecast.tsv'), sep='\t', index=False)
    normalized_test_dat_df.to_csv(os.path.join(args.output_dir, 'normalized_test_data.tsv'), sep='\t', index=False)

    predicted_data_df.to_csv(os.path.join(args.output_dir, 'predictions.tsv'), sep='\t', index=False)
    expected_data_df.to_csv(os.path.join(args.output_dir, 'expected.tsv'), sep='\t', index=False)

    with open(os.path.join(args.output_dir, 'rmse_file.tsv'), 'w') as f:
        f.write('\t'.join([str(x) for x in rmse_per_month]))
        f.write('\n' + str(total_rmse))

    mean_training_time = training_time / len(predicted_data_df)
    mean_testing_time = testing_time / len(predicted_data_df)

    with open(os.path.join(args.output_dir, 'time.tsv'), 'w') as f:
        f.write('mean_training_time\tmean_testing_time\n{training_time}\t{testing_time}\n'.format(training_time=mean_training_time, testing_time=mean_testing_time))
