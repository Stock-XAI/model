from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import pandas as pd

BASE_REPO = "capston-team-5/finma-pruned50-fp16"
LORA_PATH = "./finma-pruned50-lora"  # 학습 결과 저장된 경로
TOKEN     = "hf_"

# 1. 4-bit 모델 로드 (LoRA 적용 전)
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_REPO,
    token=TOKEN,
    quantization_config=bnb_cfg,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(BASE_REPO)
tokenizer.pad_token = tokenizer.eos_token
base_model.config.pad_token_id = tokenizer.pad_token_id

# 2. LoRA 가중치 로드
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

# 3. 추론 함수 정의
def generate_response(prompt, max_new_tokens=128):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# 4. 테스트 예시
import yfinance as yf
from datetime import datetime, timedelta

# 1. 날짜 범위 설정
ticker = "005930.KS"
end_date = datetime(2025, 4, 30)
start_date = end_date - timedelta(weeks=2)  # 주말 포함 넉넉하게

# 2. yfinance로 데이터 다운로드
data = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"),
                   end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"))

# 3. 필요한 컬럼 선택 및 등락률 계산
data = data[["Open", "High", "Low", "Close", "Volume"]]
data.reset_index(inplace=True)
data["Date"] = data["Date"].dt.strftime("%Y-%m-%d")
data["Change"] = data["Close"].pct_change().fillna(0)

# 4. 최근 5일만 추출
last_10 = data.tail(10)

# 5. 프롬프트 생성
context = "\n".join([
    f"{row['Date']}, {int(row['Open'].iloc[0]) if isinstance(row['Open'], pd.Series) else int(row['Open'])}, "
    f"{int(row['High'].iloc[0]) if isinstance(row['High'], pd.Series) else int(row['High'])}, "
    f"{int(row['Low'].iloc[0]) if isinstance(row['Low'], pd.Series) else int(row['Low'])}, "
    f"{int(row['Close'].iloc[0]) if isinstance(row['Close'], pd.Series) else int(row['Close'])}, "
    f"{int(row['Volume'].iloc[0]) if isinstance(row['Volume'], pd.Series) else int(row['Volume'])}, "
    f"{float(row['Change'].iloc[0]) if isinstance(row['Change'], pd.Series) else float(row['Change']):.16f}"
    for _, row in last_10.iterrows()
])

prompt = {
    "instruction": f"Assess the data to estimate how the closing price of 삼성전자 will change on 2025-04-30. \n"
                   f"Respond with one of the following levels based on the rate of change: \n"
                   f"Strong Rise (≥ 5%), Rise (2%–4.99%), Slight Rise (0%–1.99%), "
                   f"Slight Fall (–1.99% to 0%), Fall (–4.99% to –2%), or Strong Fall (≤ –5%).\n\n"
                   f"Context: date, open, high, low, close, volume, change.\n{context}\nAnswer:",
    "output": ""  # 모델 추론 시 자동 생성
}

import json
# print(json.dumps(prompt, indent=2, ensure_ascii=False))

# 모델에게 실제로 프롬프트를 던져 추론 실행
response = generate_response(prompt["instruction"])

# "Answer:" 이후 텍스트만 추출 (대소문자 구분 유의)
answer_prefix = "Answer:"
if answer_prefix in response:
    final_output = response.split(answer_prefix, 1)[-1].strip()
else:
    final_output = response.strip()  # fallback

print("\n[Model Output]")
print(final_output)