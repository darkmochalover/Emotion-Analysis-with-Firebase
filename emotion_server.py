from flask import Flask, jsonify
from flask_restful import Api
import torch
# from model.kogpt2 import DialogKoGPT2
from transformers import BertTokenizerFast
from tensorflow import keras
from transformers import PreTrainedTokenizerFast


#모델 gpu 사용 여부
ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

#모델 구조 불러오기
model = keras.models.load_model('modelsave/emotion_detect.h5')
tokenizer = PreTrainedTokenizerFast(tokenizer_file='modelsave/emotion_tokenizer.json')

#학습한 모델 불러오기
# model.load_state_dict(checkpoint['model_state_dict'])
# model.summary()

# Flask 인스턴스 정리

app = Flask(__name__)

api = Api(app)

@app.route('/echo_call/<param>') #get echo api
def get_echo_call(param):
    # param : 입력으로 들어오는 텍스트임
    print(param)

    # 입력값을 토큰화
    tokenized_indexs = tokenizer.encode(param)
    # 모델에 input 으로 넣기 위한 shape 으로 변환
    input_ids = torch.tensor([tokenizer.bos_token_id, ] + tokenized_indexs + [tokenizer.eos_token_id]).unsqueeze(0)
    # 모델에 넣고, 예측 값을 sample_output 에 저장
    sample_output = model.generate(input_ids=input_ids)
    # 디코딩을 통해서 안드로이드에 전달할 텍스트로 변환
    ans = tokenizer.decode(sample_output[0].tolist()[len(tokenized_indexs) + 1:], skip_special_tokens=True)
    ans = ans[:ans.find(".")]

    print(ans)

    return jsonify({"param": ans}) # 모델의 예측 결과를 JSON 형태로 반환

# 서버를 실행 할 host 와 port 를 지정
if __name__ == '__main__':
    app.run(host='172.16.232.193',port=5000,debug=True)