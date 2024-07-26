from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import joblib

app = Flask(__name__)
CORS(app)  # app.py에 CORS 적용

data = pd.read_csv('pre_J24.csv')
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

def create_sequences(features, input_seq_length):
    xs = []
    for i in range(len(features) - input_seq_length + 1):
        x = features[i:i+input_seq_length]
        xs.append(x)
    return np.array(xs)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

input_size = 3
num_layers = 3
hidden_size = 64
output_size = 24

# 디바이스 설정 (GPU가 사용 가능한 경우 GPU, 그렇지 않으면 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드
model = torch.load('LSTM_24.pth', map_location=device)
model.to(device)
model.eval()  # 모델 평가 모드로 설정

# 스케일러 로드
ft_scaler = joblib.load('ft24_scaler.pkl')
tg_scaler = joblib.load('tg24_scaler.pkl')

def predict(date_str):
    try:
        # 입력 받은 날짜와 시간
        input_date = datetime.strptime(date_str, '%Y%m%d%H%M%S')
    except ValueError:
        return "Invalid date format"

    # 과거 48시간 데이터 추출
    start_date = input_date - timedelta(hours=48)
    end_date = input_date - timedelta(hours=1)
    past_data = data.loc[start_date:end_date]

    if past_data.shape[0] != 48:
        print(f"Expected 48 rows, but got {past_data.shape[0]} rows.")
        return "Insufficient data for the specified date range"
    # 데이터 전처리
    features = ['height1', 'height2', 'in_flow']
    past_features = past_data[features]
    scaled_features = ft_scaler.transform(past_features)

    # 시퀀스 생성
    input_seq_length = 48
    X = create_sequences(scaled_features, input_seq_length)

    # 모델 예측
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        prediction = model(input_tensor).cpu().numpy()

    # 역변환하여 실제 값으로 변환
    predicted_values = tg_scaler.inverse_transform(prediction).flatten()

    return predicted_values.tolist()

    # # 예측 값에 대한 타임스탬프 생성
    # prediction_start_time = input_date + timedelta(hours=1)
    # prediction_times = [prediction_start_time + timedelta(hours=i) for i in range(24)]

    # # 예측 값과 타임스탬프를 리스트로 반환
    # predictions = [[time.strftime('%Y-%m-%d %H:%M:%S'), value] for time, value in zip(prediction_times, predicted_values.tolist())]
    # return predictions

# 예측 API 엔드포인트 정의
@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.get_json()
    date_str = data.get('date')
    
    if not date_str:
        return jsonify({'error': 'No date provided'}), 400
    
    try:
        predictions = predict(date_str)
        if isinstance(predictions, str):  # 에러 메시지일 경우
            return jsonify({'error': predictions}), 400
        return jsonify({'predictions': predictions}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)