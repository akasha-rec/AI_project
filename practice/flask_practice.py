from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import pandas as pd
import joblib
import datetime

app = Flask(__name__)
CORS(app)  # app.py에 CORS 적용

# 모델 로드
model = torch.load('LSTM.pth')
model.eval()  # 모델 평가 모드로 설정

# 디바이스 설정 (GPU가 사용 가능한 경우 GPU, 그렇지 않으면 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # 모델을 디바이스로 이동

# 스케일러 로드
ft_scaler = joblib.load('ft_scaler.pkl')
tg_scaler = joblib.load('tg_scaler.pkl')

# 데이터 로드
j = pd.read_csv('data/J(배).csv', header=0)
j['date'] = pd.to_datetime(j['date'])
j.set_index('date', inplace=True)

def create_sequences(features, seq_length):
    xs = []
    for i in range(len(features) - seq_length + 1):
        x = features[i:i+seq_length]
        xs.append(x)
    return np.array(xs)

def predict(start_date, hours_ahead):
    # 데이터 범위 설정: start_date - 24시간 부터 start_date까지
    end_date = start_date + datetime.timedelta(hours=hours_ahead)
    start_date = start_date - datetime.timedelta(hours=24)
    print(f"Start date: {start_date}, End date: {end_date}")

    sample_data = j.loc[start_date:end_date]  # 24시간 + hours_ahead 데이터 포함
    print(f"Sample data length: {len(sample_data)}")
    
    if len(sample_data) < 24:
        return "Not enough data to make a prediction"

    sample_features = sample_data[['height1', 'height2', 'in_flow']].values
    print(f"Sample features before scaling: {sample_features}")

    sample_features = ft_scaler.transform(sample_features)
    print(f"Sample features after scaling: {sample_features}")

    # 생성된 시퀀스 중 마지막 시퀀스만 사용 (최신 데이터 기준)
    sample_sequence = create_sequences(sample_features, 24)[-1]  # 마지막 시퀀스만 선택
    print(f"Sample sequence shape: {sample_sequence.shape}")
    
    sample_sequence = torch.tensor(sample_sequence, dtype=torch.float32).unsqueeze(0).to(device)  # 배치 차원 추가 및 디바이스 이동

    with torch.no_grad():
        prediction = model(sample_sequence)
    
    prediction = prediction.cpu().numpy()  # 결과를 CPU로 이동
    prediction = tg_scaler.inverse_transform(prediction)
    print(f"Prediction before inverse scaling: {prediction}")

    return float(prediction[0][0])  # float 타입으로 변환하여 반환

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()
    start_date_str = data['start_date']
    hours_ahead = int(data['hours_ahead'])

    start_date = datetime.datetime.strptime(start_date_str, '%Y%m%d%H%M%S')
    
    prediction = predict(start_date, hours_ahead)
    
    return jsonify({'prediction': prediction})
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)