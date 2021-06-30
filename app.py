import os

from flask import Flask, flash, request, redirect, url_for, render_template, jsonify, make_response
from werkzeug.utils import secure_filename
from datetime import datetime
from flask import send_from_directory
import pandas as pd
from six.moves import urllib
import json

from forecaster.factory import PredictorsFactory

UPLOAD_FOLDER = 'files_storage'

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'tsv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.urandom(24)

factory = PredictorsFactory()

lines_predict = {}
lines_orig = []


@app.route("/data.json")
def data():
    global lines_orig

    series = [{
        'name': 'Исходные данные',
        'data': lines_orig,
        'type': 'scatter'

    }] + \
             [{
                 'name': k,
                 'data': lines_predict[k]
             } for k in lines_predict]

    return jsonify(title={
        'text': 'Прогноз'
    },
        rangeSelector={
            'selected': '1',
            'inputEnabled': False
        },
        legend={
            'enabled': True
        },
        series=series)


@app.route('/get_file')
def get_file():
    global lines_orig

    res = pd.DataFrame()
    dates = None

    for k in lines_predict:
        if dates is None:
            dates = list(map(lambda x: str(datetime.fromtimestamp(x[0] / 1000))[:10], lines_predict[k]))
            res['fielddate'] = dates
            res['real_data'] = list(map(lambda x: x[1], lines_orig)) + [None] * (len(dates) - len(lines_orig))
        res[k] = list(map(lambda x: x[1], lines_predict[k]))

    res.to_csv('result.csv', index=False)
    with open('result.csv', 'rb') as f:
        file = f.read()

    response = make_response(file)
    response.headers.set('Content-Type', 'csv')
    response.headers.set('Content-Disposition', 'attachment', filename='result.csv')
    return response


@app.route('/')
def index():
    return render_template('index.html', plot_data=False)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def np_to_datetime(x):
    return datetime.utcfromtimestamp(x.tolist() / 1e9)


def get_dates_and_targets(df, ylabel='target'):
    X, y = None, None
    for col in df.columns:
        try:
            X = list(map(lambda x: np_to_datetime(x),
                         df[col].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d')).values))
            y = df[ylabel].values
            break
        except ValueError:
            continue

    if X is None:
        raise ValueError('Столбец даты не обнаружен!')

    return X, y


def datetime_to_ms(k):
    return int(k.timestamp() * 1000 + 10_800_000)


@app.route("/get_predictions", methods=['POST'])
def get_predictions():
    global lines_orig
    global lines_predict

    if 'data_file' not in request.files:
        flash('Нет требуемого файла!')
        return redirect(request.url)

    file = request.files['data_file']
    date_to = request.form['date_to']
    date_to = datetime.strptime(date_to, '%Y-%m-%d')
    selector = request.form['select']

    if file.filename == '':
        flash('Не выбран файл!')
        return redirect(request.url)

    if not allowed_file(file.filename):
        flash('Неверный тип файла, поддерживаются только ' + ', '.join(ALLOWED_EXTENSIONS))
        return redirect(request.url)

    filename = file.filename  # secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    try:
        df = get_dataframe(filename)
        x, y = get_dates_and_targets(df)
        lines_orig = list(zip(list(map(lambda k: datetime_to_ms(k), x)),
                              list(map(lambda k: float(k), y))))

        if selector == 'any':
            types = factory.get_all_methods()
        else:
            types = [selector]

        for mod in types:
            model = factory.get_model(mod)
            predictions = model.fit_predict(x, y, date_to)
            dates = list(range(min(list(map(lambda k: datetime_to_ms(k), x))),
                               datetime_to_ms(date_to) + 1, 86_400_000))
            lines_predict[mod] = list(zip(dates, predictions))

    except ValueError as e:
        flash(e.args[0])
        return render_template('index.html', plot_data=False)

    return render_template('index.html',
                           plot_data=True,
                           date_to=request.form['date_to'],
                           select=request.form['select'])


def get_dataframe(filename):
    filename_full = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        if filename.rsplit('.', 1)[1].lower() == 'csv':
            df = pd.read_csv(filename_full)
        elif filename.rsplit('.', 1)[1].lower() in ('xls', 'xlsx'):
            df = pd.read_excel(filename_full)
        elif filename.rsplit('.', 1)[1].lower() == 'tsv':
            df = pd.read_csv(filename_full, sep='\t')
        else:
            df = pd.DataFrame()
    except IndexError:
        raise ValueError(f'Недопустимое название файла: {filename}')
    return df


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
