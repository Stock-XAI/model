from transformers import LlamaTokenizer, LlamaForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import Dataset
import json
import torch
import os
from huggingface_hub import login

login()

# ✅ 환경 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ✅ 모델 & 토크나이저 로드
model_name = "ChanceFocus/finma-7b-full"
token = "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"
tokenizer = LlamaTokenizer.from_pretrained(model_name, token=token)

# ✅ 4bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = LlamaForCausalLM.from_pretrained(
    model_name,
    token=token,
    device_map="auto",
    quantization_config=bnb_config
)

# ✅ LoRA 준비
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,  # 메모리 절약
    bias="lora_only",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ✅ 패딩 토큰 지정
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# ✅ 데이터 로딩
def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]

raw_data = load_jsonl("output.jsonl")
dataset = Dataset.from_list(raw_data)

# ✅ 프롬프트 포맷
def format_prompt(example):
    return {"text": f"{example['instruction']}\nAnswer: {example['output']}"}

formatted_dataset = dataset.map(format_prompt)

# ✅ 토크나이즈 (짧게)
def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = formatted_dataset.map(tokenize)

# ✅ 학습 설정 (4070 SUPER 최적화)
training_args = TrainingArguments(
    output_dir="./finma-lora-stock",
    per_device_train_batch_size=1,  # ✅ VRAM 12GB 기준 안전값
    gradient_accumulation_steps=8,  # 배치 사이즈 효과적으로 키움
    num_train_epochs=8,
    learning_rate=2e-4,
    fp16=True,
    save_strategy="epoch",
    logging_steps=10,
    report_to="none",
    gradient_checkpointing=True
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

trainer.train()

# ✅ 저장
model.save_pretrained("./finma-lora-stock")
tokenizer.save_pretrained("./finma-lora-stock")