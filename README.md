# Stock-XAI: Financial Forecasting with Explainable AI

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/Stock-XAI/model)

## 📌 Overview

This repository contains the codebase and configuration files for the `finma-7b-4bit-quantized` model, a quantized and LoRA-finetuned version of a large language model applied to financial time series forecasting. The goal of this project is to enable resource-efficient financial prediction while enhancing interpretability through Explainable AI (XAI) methods.

---

## 🚀 Project Goals

### ✅ Accomplished (Past 2 Weeks)

* **Model Size Reduction:**
  Compressed `finma-7b-full` to 3.87GB using 4-bit quantization.
  🔗 [View Model on Hugging Face](https://huggingface.co/capston-team-5/finma-7b-4bit-quantized)

* **Training Optimization:**
  Applied LoRA fine-tuning and memory-efficient training configurations.

### 🔄 Ongoing (Current Cycle)

* **SHAP Integration:**
  Applying SHAP (SHapley Additive exPlanations) to understand feature contributions.

* **XAI Exploration:**
  Comparing additional explainability techniques (e.g., Integrated Gradients, Attention Rollout).

* **Regression Finetuning:**
  Adapting the model for regression tasks to improve stock price forecasting accuracy.

---

## 🧠 Model Details

* **Base Model:** finma-7b-full
* **Quantization:** 4-bit (GPTQ)
* **Fine-Tuning:** LoRA (Low-Rank Adaptation)
* **Frameworks:** Transformers, PEFT, BitsAndBytes

---

## 📊 Dataset

The dataset consists of daily KOSPI stock prices for the top-listed companies in South Korea, including:

* Open, High, Low, Close, Volume
* Daily % Change
