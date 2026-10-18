# Demo video script (about 5 minutes)

**Before recording:** sign into Hugging Face, open the Space, wait until it says **Running**.  
Live app: https://huggingface.co/spaces/Emmanuel-NS/farumasi-pharmacist-qa  
GitHub: https://github.com/Emmanuel-NS/FARUMASI-AI

Read the **SAY** lines. Do the **DO** actions when you see them.

---

### 1. Intro (20 sec)

**SAY:**  
Hi, I am Emmanuel Nsabagasani. This is my NLP summative: FARUMASI Pharmacist Assistant. It answers routine medicine questions in English using trusted leaflet text. I will cover the problem, model, experiments, evaluation, and a live demo.

---

### 2. Problem (40 sec)

**SAY:**  
I built this because of FARUMASI, my digital pharmacy platform in Rwanda. Pharmacists are busy, but patients keep asking the same basic questions: what is this for, side effects, how do I take it. A normal chatbot can invent doses. My system only answers from trusted sources, refuses when it does not know, and sends emergencies to SAMU 912. Language is English. Users are patients and caregivers. It is not a doctor.

---

### 3. Data (25 sec)

**DO:** Open GitHub README, Data table.

**SAY:**  
Data comes from openFDA, MedQuAD, MedicationQA, Rwanda FDA leaflets, and CLINC150. About 8,800 passages and 9,500 questions. Splits are grouped by medicine so nothing leaks. Limitation: some Rwanda PDFs were scanned and could not be used.

---

### 4. Model (60 sec) — important for marks

**DO:** Show the system overview in the README.

**SAY:**  
The pipeline has five parts.  
One: rules for greetings and emergencies.  
Two: DistilBERT classifies intent: medicine, condition, or out of scope.  
Three: I expand local names, like panadol to paracetamol.  
Four: MiniLM retrieves the best leaflet passages.  
Five: Qwen 1.5B with LoRA writes the answer from those passages only.  

LoRA means I train small adapters, not the full model, on free Colab. Key settings: LoRA rank 16, one epoch, learning rate 2e-4, loss on answer tokens only. I also trained Flan-T5 LoRA as a backup.

---

### 5. Experiments (40 sec) — important for marks

**DO:** Show Results tables in README.

**SAY:**  
Baselines: TF-IDF for intent, BM25 for retrieval, and zero-shot Qwen for answers.  

What I learned: formal test accuracy looked perfect, but real patient questions failed until I added consumer-style training data.  

Fine-tuned MiniLM more than doubled BM25 Recall at 1.  

Qwen with LoRA raised ROUGE-L from 0.327 to 0.643, support from 0.62 to 0.96, and abstention from 25% to 73%. So fine-tuning made answers more grounded and safer.

---

### 6. Metrics and errors (20 sec)

**SAY:**  
I used accuracy and F1 for intent, Recall and MRR for retrieval, ROUGE-L and BERTScore for answers, plus support and abstention for grounding. These are not clinical scores. Next I show live success and failure cases.

---

### 7. Live demo (2 min) — important for marks

**DO:** Open the Space chat.

**SAY:**  
This Gradio app calls my trained models on ZeroGPU and shows the answer plus sources.

**DO:** Type `What are the side effects of metformin?`  
**SAY:** Success. Side effects from the trusted label, with sources below.

**DO:** Type `after sex pills`  
**SAY:** Local phrasing mapped to emergency contraception.

**DO:** Type `what do you know about quarteme`  
**SAY:** Close-name fix. It suggests Coartem instead of inventing a drug.

**DO:** Type `what do you know about kopavitamins`  
**SAY:** Failure handled honestly. Product not in sources. Better than a fake answer.

**DO:** Type `what is the exchange rate of the rwandan franc?`  
**SAY:** Out of scope. Intent model refuses.

---

### 8. Close (20 sec)

**DO:** Show GitHub repo quickly.

**SAY:**  
Code, notebooks, and results JSON are on GitHub for reproducibility. I designed the pipeline, ran training, and tested the live app. Limits: English only, not medical advice. Goal: fewer repeated questions for busy pharmacies, safer first answers for patients. Thank you.

---

## If they ask (one-line answers)

| Question | Answer |
|---|---|
| Why not ChatGPT? | It can invent doses; my brief requires a trained, grounded model. |
| What is LoRA? | Small adapters on a frozen model so I can fine-tune on free Colab. |
| Best result? | Qwen LoRA: ROUGE-L 0.643 vs 0.327 zero-shot. |
| Medical advice? | No. Disclaimer on every answer. Emergencies: call 912. |
