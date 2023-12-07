from typing import Dict, List
import tensorflow as tf
import numpy as np
import pandas as pd

class MyModel:
    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)

    @staticmethod
    def help():
        print("딕셔너리 형태로 입력해야 합니다.\n면적당보증금,면적당매매금,전세율,위도,경도,이자율을 \n5일치씩 입력해야 합니다")

    def input_data(self, data: Dict[str, List[float]]) -> np.ndarray:
        columns = ['면적당보증금','면적당매매금','전세율','위도','경도','이자율']
        df = pd.DataFrame(columns=columns)

        df = pd.DataFrame.from_dict(data, orient='columns')
        for column in columns:
            if column in data:
                df[column] = data[column]

        pred = np.expand_dims(df.to_numpy(), axis=0)  # If you want to display the DataFrame
        return pred

    def model_2(self, data: np.ndarray):
        try:
            pred = self.model.predict(data, verbose=0)
            pred = np.where(pred > 0.5, 1, 0)[0][0]
            return pred
        except:
            return 'error'

#예제

my_model = MyModel(model_path='simple_model_3.h5')

data = {
    '면적당보증금': [207.164427, 199.646793, 222.682534, 260.329239, 220.911840],
    '면적당매매금': [228.137412, 223.839083, 248.238090, 248.297844, 236.294295],
    '전세율': [90.806863, 89.192107, 89.705224, 104.845550, 93.490129],
    '위도': [37.772679, 37.772679, 37.772679, 37.772679, 37.772679],
    '경도': [128.943061, 128.943061, 128.943061, 128.943061, 128.943061],
    '이자율': [0.0285, 0.0264, 0.0269, 0.0274, 0.0294]
}

input_data = my_model.input_data(data)
prediction = my_model.model_2(input_data)
print(f"Prediction: {prediction}")
