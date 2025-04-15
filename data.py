from pymongo import MongoClient
import json
from collections import defaultdict
from datetime import datetime

def classify_change(rate):
    if rate >= 0.05:
        return "Strong Rise (≥ 5%)"
    elif rate >= 0.02:
        return "Rise (2%–4.99%)"
    elif rate >= 0.0:
        return "Slight Rise (0%–1.99%)"
    elif rate > -0.02:
        return "Slight Fall (–1.99% to 0%)"
    elif rate > -0.05:
        return "Fall (–4.99% to –2%)"
    else:
        return "Strong Fall (≤ –5%)"

# 1. MongoDB 연결
uri = f"mongodb+srv://skkucapstone:team5@stock.iz5b97b.mongodb.net/?retryWrites=true&w=majority&appName=stock"
client = MongoClient(uri)
db = client["stock"]
collection = db["kospi_top50_2024"]

company_week_data = defaultdict(list)
answer = {}

cursor = collection.find({})

# 2. JSONL 문자열 만들기

for doc in cursor:
    date = doc.get("Date")
    open_price = doc.get("Open")
    high_price = doc.get("High")
    low_price = doc.get("Low")
    close_price = doc.get("Close")
    volume = doc.get("Volume")
    change = doc.get("Change")
    name = doc.get("Name")

    if None in (date, open_price, close_price, name):
        continue

    # 날짜 파싱 및 ISO 주차 추출
    try:
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        year_key, week_key, _ = date_obj.isocalendar()
    except Exception as e:
        print(f"날짜 파싱 오류: {date}")
        continue

    key = (name, year_key, week_key)

    line = f"{date}, {open_price}, {high_price}, {low_price}, {close_price}, {volume}, {change}"
    company_week_data[key].append(line)

    if key not in answer:
        answer[key] = change  # 기준 주차의 change

# 예외 처리용 강제 추가 예시 (있으면 유지, 없으면 제거 가능)
answer[('삼성전자', 2025, 1)] = '0.003759'

jsonl_str = ""
for (company, year, week), lines in company_week_data.items():
    # 다음 주차 계산 (단순 +1 → 연말/연초 cross는 정확히 계산하려면 더 복잡함)
    next_year = year
    next_week = week + 1
    if next_week > 52:
        next_year += 1
        next_week = 1

    prefix = f"Assess the data to estimate how the closing price of {company} will change in week {next_week} of {next_year}. Respond with one of the following levels based on the rate of change: Strong Rise (≥ 5%), Rise (2%–4.99%), Slight Rise (0%–1.99%), Slight Fall (–1.99% to 0%), Fall (–4.99% to –2%), or Strong Fall (≤ –5%). Context: date, open, high, low, close, volume, change."
    query = prefix + "\n" + "\n".join(lines) + "\nAnswer:"

    answer_key = (company, next_year, next_week)
    if answer_key not in answer:
        continue  # 예측 주차 데이터가 없으면 건너뜀

    answer_change = float(answer[answer_key])
    json_obj = {
        "instruction": query,
        "output": classify_change(answer_change)
    }
    jsonl_str += json.dumps(json_obj, ensure_ascii=False) + "\n"

# 3. 파일로 저장
with open("output.jsonl", "w", encoding="utf-8") as f:
    f.write(jsonl_str)

print("✅ 저장 완료: output.jsonl")
