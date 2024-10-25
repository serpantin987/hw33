
import os
import dill
from glob import glob
import pandas as pd
import json

path = os.environ.get('PROJECT_PATH', '/opt/airflow')


def predict():
    model_last = sorted(os.listdir(f'{path}/data/models/'))[-1]
    with open(f'{path}/data/models/{model_last}', 'rb') as file:
        model = dill.load(file)
    data = pd.DataFrame(columns=['id','pred','price'])
    for datapath in glob(f'{path}/data/test/*.json'):
        with open(datapath, 'rb') as datafile:
            df = pd.json_normalize(json.load(datafile))
            y = model.predict(df)
            dat = {
                    'id': df.id,
                    'predict': y[0],
                    'price': df.price
             }
            df_data = pd.DataFrame(dat)
            df_pred = pd.concat([data, df_data], axis=0)
    df_pred.to_csv(f'{path}/data/predictions/cars_pipe_predict.csv')


if __name__ == '__main__':
    predict()
