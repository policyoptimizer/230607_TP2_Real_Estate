import streamlit as st
import pandas as pd
from class_model import MyModel
from typing import Dict, List
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np



file_path = "학습데이터.csv"





@st.cache_data
def load_data():
  df = pd.read_csv(file_path)
  location_dict = {}

  for index, row in df.iterrows():
    location_dict[row['시군구']] = {'위도': row['위도'], '경도': row['경도']}
  return df,location_dict




my_model = MyModel(model_path='simple_model_3.h5')
df,location_dict = load_data()

st.title("시군구별 역전세 예측")
st.subheader('입력한 연월과 위치정보를 기반으로 1개월 후의 역전세를 예측합니다', divider='rainbow')
year = st.selectbox('연도를 선택하세요.', list(range(2023,2024)))
month = st.selectbox('월을 선택하세요.', list(range(6, 12)))
target_date = int(f"{year}{month:02d}") -1





if target_date >= 202308:
    st.warning('선택하신 연월은 2023년 8월보다 클 수 없습니다.')
    st.stop()

  



# '시군구' 컬럼을 공백으로 쪼개서 새로운 DataFrame 생성
split_df = df['시군구'].str.split(' ', expand=True)
split_df.columns = ['도', '시군', '읍면동', '리']

# Streamlit 앱
# Streamlit 앱
col1, col2, col3, col4 = st.columns(4)

with col1:
    selected_do = st.selectbox('도를 선택하세요.', options=split_df['도'].fillna("").unique())
    filtered_df_by_do = split_df[split_df['도'] == selected_do]

with col2:
    if not filtered_df_by_do.empty:
        selected_sigun = st.selectbox('시/군을 선택하세요.', options=filtered_df_by_do['시군'].fillna("").unique())
        filtered_df_by_sigun = filtered_df_by_do[filtered_df_by_do['시군'] == selected_sigun]
    else:
        selected_sigun = None

with col3:
    if selected_sigun:
        selected_eup = st.selectbox('읍/면/동을 선택하세요.', options=filtered_df_by_sigun['읍면동'].fillna("").unique())
        filtered_df_by_eup = filtered_df_by_sigun[filtered_df_by_sigun['읍면동'] == selected_eup]
    else:
        selected_eup = None

with col4:
    if selected_eup:
        selected_ri = st.selectbox('리를 선택하세요.', options=filtered_df_by_eup['리'].fillna("").unique())

input_data = str(selected_do + " "+ selected_sigun + " "+selected_eup  +" "+ selected_ri).rstrip()


matched_indices = list(df.query("`계약년월` == @target_date and `시군구` == @input_data").index)[0]


if matched_indices is not None and 0 <= matched_indices < len(df):

    # 인덱스가 음수가 되거나, DataFrame의 길이를 초과하지 않도록 조정
    start_index = max(0, matched_indices - 4)
    end_index = min(len(df) -1, matched_indices)  # DataFrame의 마지막 인덱스를 넘지 않도록

    input_dict = {}
    try:
        for col in ['면적당보증금', '면적당매매금', '전세율', '위도', '경도', '이자율']:
            input_dict[col] = list(df[col].iloc[start_index:end_index+1])
    except pd.errors.InvalidIndexError:
        st.write("인덱스가 유효한 범위를 벗어났습니다.")

else:
    st.write("유효하지 않은 matched_indices 값입니다.")


input_data = my_model.input_data(input_dict)
prediction = my_model.model_2(input_data)

if prediction == 0:
    st.markdown(f"<span style='color: blue;'>{year}년 {month}월에 전세율이 80퍼센트를 넘지 않을 것으로 예상됩니다</span>", unsafe_allow_html=True)
else:
    st.markdown(f"<span style='color: red;'>{year}년 {month}월에 전세율이 80퍼센트를 넘을 것으로 예상됩니다</span>", unsafe_allow_html=True)


new_dict = {
    

    '연월' : range(target_date-4,target_date+1),
    '전세율' : input_dict['전세율']

}


# 데이터를 Pandas DataFrame으로 변환하고 인덱스를 설정
df = pd.DataFrame(new_dict)
df['연월'] = pd.to_datetime(df['연월'], format='%Y%m')  # 연월을 datetime 타입으로 변환
df.set_index('연월', inplace=True)

# Streamlit에서 라인 차트 그리기
st.line_chart(df)