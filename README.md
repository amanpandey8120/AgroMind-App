# AgroMind 🌱

**AgroMind** is a fine-tuned Large Language Model built for the **Agro Aid AI App**. It analyzes soil report parameters (N, P, K, pH, EC, micronutrients, climate data) and generates natural-language guidance on crop selection, fertilizer recommendation, irrigation timing, and pesticide usage — without relying on any third-party LLM API (e.g. ChatGPT/OpenAI).

---

## Overview

AgroMind is built by fine-tuning **Qwen2.5-3B-Instruct** on an agriculture-specific instruction dataset using **LoRA/QLoRA** (via Unsloth + TRL). The goal is to give farmers and agronomy users direct, private, self-hosted AI guidance based on their soil report data.

---

## Features

- 📊 Soil-report-based crop recommendation
- 🌾 Fertilizer recommendation based on soil nutrients
- 💧 Irrigation guidance *(planned — pending advisory dataset integration)*
- 🐛 Pesticide/chemical usage guidance *(planned — pending advisory dataset integration)*
- 💬 Conversational chat interface (Alpaca-style instruction tuned)
- 🔒 Fully self-hosted — no external LLM API dependency

---

## Model Details

| | |
|---|---|
| **Base model** | Qwen2.5-3B-Instruct |
| **Fine-tuning method** | LoRA / QLoRA (parameter-efficient fine-tuning) |
| **Framework** | Unsloth + Hugging Face TRL (`SFTTrainer`) |
| **Trainable parameters** | ~30M (0.96% of total 3.1B) |
| **Precision** | bf16 / fp16 (auto-selected based on GPU support) |
| **Data format** | Alpaca-style (`instruction`, `input`, `output`) |

---

## Dataset

Training data was compiled from:
- Kaggle crop recommendation and soil-NPK datasets
- Kaggle fertilizer recommendation dataset
- Converted into Alpaca instruction format

**Note:** Current dataset produces single-label outputs (crop/fertilizer name). Conversational explanatory outputs and ICAR/FAO/state agri-advisory text (irrigation schedules, pesticide dosage guidance) are planned additions — see [Roadmap](#roadmap).

Dataset format example:
```json
{
  "instruction": "Based on the given soil and environmental parameters, predict the correct crop, fertilizer, or label.",
  "input": "nitrogen: 143\nphosphorus: 69\npotassium: 217\nph: 5.9\nec: 0.58\nsulfur: 0.23\ncopper: 10.2\niron: 116.35\nmanganese: 59.96\nzinc: 54.85\nboron: 21.29",
  "output": "pomegranate"
}
```

---

## Repository Structure

```
AgroMind/
├── data/
│   ├── raw/                     # Original source CSVs
│   ├── processed/               # Cleaned, standardized, deduplicated data
│   └── alpaca_dataset.jsonl     # Final instruction-formatted training data
├── notebooks/
│   ├── 01_data_preparation.ipynb    # Inspect, clean, standardize, convert
│   └── 02_finetune_qwen.ipynb       # LoRA/QLoRA fine-tuning notebook
├── models/
│   └── qwen2.5_agri_lora/       # Saved LoRA adapters / checkpoints
├── inference/
│   └── generate.py              # Script to load model and run inference
├── app/                          # Agro Aid App backend integration code
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/AgroMind.git
cd AgroMind
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

Key dependencies:
```
torch
transformers
trl
unsloth
peft
bitsandbytes
datasets
accelerate
```

### 3. (Optional) Set Hugging Face token
```bash
export HF_TOKEN=your_huggingface_token
```

---

## Training

Training is done via `notebooks/02_finetune_qwen.ipynb` using Unsloth + TRL's `SFTTrainer` with LoRA.

Key settings used:
- Base model: `Qwen2.5-3B-Instruct`
- Batch size: 1 (gradient accumulation: 8 → effective batch size 8)
- Learning rate: `2e-4`, cosine schedule
- Epochs: 1 (adjustable)
- Optimizer: `paged_adamw_8bit`
- Checkpointing: every 200 steps, saved to Google Drive for crash/disconnect recovery

To resume training after an interruption, simply re-run the training cell — it automatically detects and resumes from the latest valid checkpoint.

---

## Inference

```bash
python inference/generate.py --input "nitrogen: 90, phosphorus: 42, potassium: 43, ph: 6.5, rainfall: 200mm"
```

Or programmatically:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("models/qwen2.5_agri_lora")
tokenizer = AutoTokenizer.from_pretrained("models/qwen2.5_agri_lora")

prompt = "My soil report shows: Nitrogen 90, Phosphorus 42, Potassium 43, pH 6.5. Which crop should I grow?"
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## Deployment

AgroMind is deployed as a self-hosted inference service (no external LLM API calls):

- Model weights hosted on a GPU cloud instance (e.g. RunPod, AWS, GCP)
- Served via a lightweight inference API (e.g. FastAPI + `transformers`/`vLLM`)
- Agro Aid App backend calls this internal endpoint instead of ChatGPT/OpenAI API

```bash
# Example: run inference server
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

---

## Roadmap

- [ ] Expand outputs from single-word labels to full explanatory responses
- [ ] Integrate ICAR / KVK / state agri-department advisory content (irrigation schedules, pesticide guidance)
- [ ] Add FAO and data.gov.in datasets via RAG pipeline for up-to-date, sourced answers
- [ ] Add multi-turn conversational fine-tuning examples
- [ ] Add regional language support (Hindi and other Indian languages)
- [ ] Quantize model (GGUF) for lightweight/edge deployment

---

## Disclaimer

AgroMind provides AI-generated agricultural guidance based on available training data. It is **not a replacement for professional agronomist advice**, especially regarding pesticide/chemical dosages. Always verify critical recommendations (especially chemical usage) with local agriculture extension officers before application.

---

## License

*(Specify your license here — e.g. MIT, Apache 2.0, or proprietary)*

## Acknowledgements

- Base model: [Qwen2.5](https://github.com/QwenLM/Qwen2.5) by Alibaba Cloud
- Fine-tuning framework: [Unsloth](https://github.com/unslothai/unsloth), [Hugging Face TRL](https://github.com/huggingface/trl)
- Data sources: Kaggle open datasets, ICAR, FAO, data.gov.in
