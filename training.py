from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import load_dataset
import torch, os
from huggingface_hub import login

login()

# capstone-team-5/finma-pruned50-fp16
# ChanceFocus/finma-7b-full
BASE_REPO   = "ChanceFocus/finma-7b-full"
TOKEN       = "hf_"
LORA_R      = 64
LORA_ALPHA  = 64
LORA_DROPOUT= 0.1
MAX_LEN     = 256
BATCH_SZ    = 4
NUM_EPOCH   = 3
LR          = 2e-4
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ================================================================
# 1) 4-bit NF4 모델 로드
# ================================================================
bnb_cfg = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.float16)

model = AutoModelForCausalLM.from_pretrained(
    BASE_REPO,
    token=TOKEN,
    quantization_config=bnb_cfg,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(BASE_REPO)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# ================================================================
# 2) LoRA 삽입
# ================================================================
model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    bias="lora_only",
    target_modules=["q_proj", "v_proj", "o_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ================================================================
# 3) 데이터 로드 & 전처리
#    (sample: 5-day KOSPI jsonl  →  train / test 9:1 split)
# ================================================================
DATA_URL = "https://raw.githubusercontent.com/Stock-XAI/Data/refs/heads/main/kospi_daily_output_10_days.jsonl"
raw_ds   = load_dataset("json", data_files={"data": DATA_URL})["data"]
split    = raw_ds.train_test_split(test_size=0.1, seed=42)
train_ds, test_ds = split["train"], split["test"]

def format_prompt(ex):
    return {"text": f"{ex['instruction']}\nAnswer: {ex['output']}"}
train_ds = train_ds.map(format_prompt, remove_columns=train_ds.column_names)
test_ds  = test_ds .map(format_prompt, remove_columns=test_ds.column_names)

def tokenize(batch):
    tok = tokenizer(batch["text"],
                    max_length=MAX_LEN,
                    truncation=True,
                    padding="max_length")
    tok["labels"] = tok["input_ids"].copy()
    return tok
train_tok = train_ds.map(tokenize, batched=True, batch_size=64, remove_columns=["text"])
test_tok  = test_ds .map(tokenize, batched=True, batch_size=64, remove_columns=["text"])

# ================================================================
# 4) Trainer 세팅 & 학습
#    * remove_unused_columns=False  ← 에러 해결
# ================================================================
# args = TrainingArguments(
#     output_dir="finma-pruned50-lora",
#     per_device_train_batch_size=BATCH_SZ,
#     per_device_eval_batch_size=BATCH_SZ,
#     gradient_accumulation_steps=4,
#     num_train_epochs=NUM_EPOCH,
#     learning_rate=LR,
#     fp16=False,
#     bf16=True,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     report_to="none",
#     remove_unused_columns=False,          # ★ 중요
#     label_names=["labels"]                # ★ PeftModel 용
# )

args = TrainingArguments(
    output_dir="finma-pruned50-lora",
    per_device_train_batch_size=BATCH_SZ,
    per_device_eval_batch_size=BATCH_SZ,
    gradient_accumulation_steps=4,
    num_train_epochs=NUM_EPOCH,
    learning_rate=2e-4,
    lr_scheduler_type = "cosine",
    bf16=True,  # GPU가 지원하는 경우 (4070 이상)
    save_strategy="no",
    logging_strategy="steps",
    logging_steps=200,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    warmup_ratio=0.03,
    remove_unused_columns=False,  # ★ 중요 (PeftModel에 필요)
    label_names=["labels"]
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model           = model,
    args            = args,
    train_dataset   = train_tok,
    eval_dataset    = test_tok,
    data_collator   = data_collator,
)
trainer.train()

model.save_pretrained("finma-pruned50-lora")

# (선택) 평가
print(trainer.evaluate())