from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
import os

# 모델과 토크나이저 로딩
model_path = "./my_model"  # 너의 checkpoint 경로로 변경
model = BertForSequenceClassification.from_pretrained("kuminga/korSTS")
tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-base")
model.eval()

# CUDA 사용 가능하면 모델 옮기기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Flask 앱 생성
app = Flask(__name__)
CORS(app)

# 예측 함수
def predict_score(sentence1, sentence2):
    inputs = tokenizer(
        sentence1,
        sentence2,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=128
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        score = outputs.logits.squeeze().item()

    return round(score, 3)

def score_to_softmax(score: float, anchors=[1.7, 3.0, 4.2]):
    """
    예측 점수(float)를 기반으로 softmax 분포 반환
    anchors: [오답 기준, 중립 기준, 정답 기준]
    """
    score_tensor = torch.tensor(score)
    anchor_tensor = torch.tensor(anchors)
    
    # anchor와의 거리 제곱을 음수로 바꿔 softmax 적용
    dist = -((score_tensor - anchor_tensor) ** 2)
    probs = F.softmax(dist, dim=0)

    labels = ["오답", "중립", "정답"]

    best_label = labels[torch.argmax(probs).item()]

    return {
        "오답": round(probs[0].item(), 3),
        "중립": round(probs[1].item(), 3),
        "정답": round(probs[2].item(), 3),
        "결과" : best_label
    }

    

# API 라우트 정의
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sentence1 = data.get('s1')
    sentence2 = data.get('s2')

    if not sentence1 or not sentence2:
        return jsonify({'error': '두 문장 모두 필요합니다.'}), 400

    score = predict_score(sentence1, sentence2)
    percentages = score_to_softmax(score)
    return jsonify({
        'score': score, 
        'correct' : percentages['정답'],
        'neutral' : percentages['중립'],
        'incorrect' : percentages['오답'],
        'result' : percentages['결과']
        })



# 포트번호 : railway에서 자동으로 받게 설정
port = int(os.environ.get("PORT", 3001))

# 서버 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
