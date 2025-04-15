# Fine-Tuning FinMA-7B with LoRA and 4-Bit Quantization

This project fine-tunes the [FinMA-7B](https://huggingface.co/ChanceFocus/finma-7b-full) model for instruction-based financial language modeling using **LoRA** (Low-Rank Adaptation) and **4-bit quantization**

---

##  Features

- 4-bit quantized model loading via `BitsAndBytesConfig`
- LoRA tuning with PEFT for low-memory adaptation
- Hugging Face `transformers` and `datasets` integration
- Tokenization and dataset preparation from `jsonl` instruction-answer pairs
- Saves the fine-tuned model and tokenizer locally

---

##  Project Structure

```
├── output.jsonl           # Input training data (instruction-answer format)
├── main.py                # Main training script
├── finma-lora-stock/      # Output directory for trained model and tokenizer
```

---

##  Requirements

- Python 3.10+
- PyTorch with CUDA
- Hugging Face `transformers`, `datasets`, `peft`, `bitsandbytes`, `huggingface_hub`

Install required packages:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets peft bitsandbytes huggingface_hub
```

---

##  Authentication

Login to Hugging Face Hub:

```python
from huggingface_hub import login
login()  # Enter your token when prompted
```

---

##  Training

Training is initiated via the Hugging Face `Trainer`:

```bash
python main.py
```

Key training configurations:

- **Batch size:** 1 (with gradient accumulation of 8)
- **Epochs:** 8
- **FP16:** Enabled
- **Gradient checkpointing:** Enabled
- **Model saved after each epoch**

---

##  Output

After training, the model and tokenizer are saved to:

```
./finma-lora-stock/
```

Use `from_pretrained("./finma-lora-stock")` to reload them later.

---

##  Notes

- Padding token is set to EOS token to avoid mismatch issues.
- LoRA is applied to key transformer modules: `q_proj`, `v_proj`, `k_proj`, etc.
- Model is optimized for instruction-following generation in financial contexts.

---

Feel free to modify hyperparameters or prompt formatting based on your dataset or compute budget.

