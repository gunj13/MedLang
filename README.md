# MedLang
Women's Health Companion

**MedLang** is a specialized conversational AI assistant designed to provide accurate, context-aware guidance on **Menstrual Health** and **Pregnancy/Fertility** topics.  
It employs a novel **Hybrid RAG (Retrieval-Augmented Generation) Architecture** orchestrated by **LangGraph** to seamlessly switch between internal specialized knowledge and external retrieval.

---

## 🌟 Key Features

- **Hybrid RAG Architecture:** Combines a fine-tuned LLM (for menstrual health) with a vector database (for pregnancy knowledge) in a single workflow.
- **Reason-Then-Respond Paradigm:** The model explicitly analyzes user intent and conversation history before generating an answer, improving coherence and accuracy.
- **Multilingual Support:** Handles queries in English and Indian regional languages (e.g., Hindi).
- **Performance Optimization:** Utilizes 4-bit quantization (BitsAndBytes) to efficiently run the `Menstrual-LLaMA-8B` model on resource-constrained hardware like Google Colab GPUs.

---

## 🧠 System Architecture

The core of MedLang is a **Hybrid RAG Pipeline** that intelligently classifies the user's need.  
The `Menstrual-LLaMA-8B` model acts as the **Cognitive Core**, while the external **FAISS index** provides factual grounding for pregnancy queries.

**LangGraph** orchestrates a mandatory three-step workflow for every user query:

1. **Retrieve:** The system retrieves relevant context from the Pregnancy Q&A dataset using FAISS.  
2. **Reason:** The LLM reads the query, chat history, and retrieved context to decide —  
   - Is this a menstrual question? → Rely on fine-tuned knowledge.  
   - Is this a pregnancy question? → Use RAG.  
3. **Answer:** Generates a structured response including the reasoning component.

---

## 🚀 Setup and Run Instructions

### 1. Repository Contents

| File Name | Description | Purpose |
|------------|-------------|----------|
| `app.py` | Python script containing the Streamlit application code | Main chat interface and application logic |
| `medlang_app.ipynb` | Jupyter Notebook with setup, installation, and run commands | Recommended way to run the app in Colab |
| `merged_preg_dataset.jsonl` | 1,378 Q&A pairs used as the external Pregnancy RAG Knowledge Base | Required for RAG component |
| `test_set.jsonl` | 120-query Golden Test Set used for model evaluation | Required for evaluation script |
| `eval_medlang.py` | Standalone script for running the quantitative evaluation pipeline | Measures model accuracy and RAG fidelity |

---

### 2. Environment Configuration (API Token)

To load the private fine-tuned `Menstrual-LLaMA-8B` model, you must provide a Hugging Face Access Token.

**Steps:**  
1. Get your token: Create a read-access token from your Hugging Face account.  
2. Create a `.env` file in the root directory of your Colab or local environment and add:

```bash
# .env file content
HF_TOKEN="hf_YOUR_READ_ACCESS_TOKEN_HERE"
```

---

### 3. Launching the Streamlit App (via Colab)

Run the app using the `medlang_app.ipynb` notebook in order:

1. **Install Dependencies:**
   ```bash
   !pip install -r requirements.txt
   !npm install -g localtunnel
   ```

2. **Save app.py:**
   ```bash
   %%writefile app.py
   # (App code here)
   ```

3. **Launch Streamlit and LocalTunnel:**
   ```bash
   !streamlit run app.py &>/content/logs.txt & npx localtunnel --port 8501 & curl https://loca.lt/mytunnelpassword
   ```

**Access:** Use the public URL and Tunnel Password (your Colab IP) to open the Streamlit interface.

---

## 📊 Evaluation Pipeline

The system is evaluated using `eval_medlang.py`, benchmarking against a generic `LLaMA-3` zero-shot baseline under identical 4-bit quantized conditions.

### Metrics

| Metric | Focus Area | Result |
|---------|-------------|--------|
| **Semantic Similarity (Avg.)** | Generation Quality | **0.7733** (vs. Baseline 0.7075) |
| **Retrieval Accuracy@2** | RAG Performance | **0.8571 (60/70 correct)** |
| **Inference Latency** | Efficiency | **14.39 s/query** (vs. Baseline 23.20 s/query) |

---

### Running Evaluation

To reproduce results or test new prompt versions:

1. **Stop the App:** Ensure the Streamlit app process is stopped.  
2. **Run Evaluation:**  
   ```bash
   !python eval_medlang.py
   ```

**Key Findings:** MedLang achieved a **9.3% gain in accuracy** and a **38% reduction in latency** compared to the baseline, confirming the impact of the hybrid architecture.

---
