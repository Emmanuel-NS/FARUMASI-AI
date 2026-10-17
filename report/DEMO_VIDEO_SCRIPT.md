# Demo video script (aim 6–8 minutes)

Record **signed in** to Hugging Face so ZeroGPU quota does not cut you off.
Speak to camera / screen-share the Space. This script maps to **every video-required rubric criterion**.

Live app: https://huggingface.co/spaces/Emmanuel-NS/farumasi-pharmacist-qa  
GitHub: https://github.com/Emmanuel-NS/FARUMASI-AI  
Report PDF: `report/Emmanuel_Nsabagasani_FARUMASI_Pharmacist_QA_Report.pdf`

---

## 0. Title slide (15 s)
“My name is Emmanuel Nsabagasani. This is FARUMASI Pharmacist Assistant — a grounded English QA system for medicine questions in Rwanda. I will explain the problem, data, models, experiments, evaluation, and the live web app.”

## 1. Problem, language, users (45 s) — Criterion 1
- Patients use pharmacies first; ChatGPT can invent doses.
- Task: domain-specific QA (not MT), English, target users = patients / caregivers.
- Must refuse out-of-scope and warn on emergencies (SAMU 912).

## 2. Dataset (60 s) — Criterion 1
Show README Table or report Table 1 briefly.
- openFDA, MedQuAD, MedicationQA, Rwanda FDA leaflets, CLINC150, challenge set.
- 8,840 passages, 9,518 QA, entity-grouped splits.
- Limitation: templated openFDA questions; some Rwanda PDFs scanned.

## 3. Model architecture walkthrough (2–2.5 min) — Criterion 2 (**critical**)
Open `pharmqa/pipeline.py` briefly, then draw/explain the flow:

1. **Rules:** greeting / feedback / risk detector (`safety.py`)
2. **Intent:** DistilBERT 3-class — input tokens → Transformer → softmax
3. **Query expand:** panadol→paracetamol, after-sex pills→levonorgestrel
4. **Retriever:** MiniLM bi-encoder embeddings, cosine, drug/section boost
5. **Reader:** Qwen2.5-1.5B-Instruct + LoRA (r=16); Flan-T5 LoRA fallback
6. **Output:** answer + sources + disclaimer

Say hyperparameters out loud:
- Intent: DistilBERT, max_len 64, consumer-augmented training
- Retriever: 2 epochs, batch 64, lr 2e-5
- Qwen LoRA: r=16, 1 epoch, batch 2×accum 8, lr 2e-4, answer-token loss only

Point at one training file: `pharmqa/generation/train_chat_lora.py`.

## 4. Baselines & experiments (90 s) — Criterion 3
Explain **why** each baseline:
- Intent: TF-IDF vs DistilBERT; with/without consumer questions (style vs topic)
- Retrieval: BM25 vs zero-shot MiniLM vs fine-tuned; ± Rwanda leaflets; hybrid RRF
- Generation: extractive, Flan/Qwen zero-shot vs LoRA; gold vs retrieved context

Key lesson to say: “Formal test accuracy of 99% hid failure on real consumer text until we added patient questions.”

## 5. Metrics & error analysis (90 s) — Criterion 4
- Metrics: accuracy/F1, Recall@k/MRR, ROUGE-L, BERTScore, support, abstention
- Show Table 4 numbers: Qwen LoRA ROUGE-L 0.643 vs zero-shot 0.327; support 0.96 vs 0.62
- Live success: metformin side effects; 120hrs → ulipristal
- Live failure/honest: “kopavitamins” → product not in sources
- Explain one error cause (wrong openFDA combination label / aspect mismatch)

## 6. Live web demo (2 min) — Criterion 6
On the Space, run:
1. `hey`
2. `What are the side effects of metformin?` then `and how should I take it?`
3. `after sex pills` then `are there any of 120hrs`
4. `what do you know about quarteme`
5. `what do you know about kopavitamins`
6. `what is the exchange rate of the rwandan franc?`

While answering, say: “Gradio UI → `PharmacistAssistant.answer` → GPU for Qwen generate → returns chat + sources.”

## 7. GitHub reproducibility (30 s) — Criterion 7
Show README reproduce section + `results/` JSON files + notebooks.

## 8. Ownership / close (30 s) — Criterion 8
“I designed the pipeline, ran Colab training, interpreted results, and fixed live failures. AI tools helped scaffold code; I verified every experiment. Limitations: English only, not clinical advice.”

---

## Recording tips
- 1080p, clear mic, scroll slowly
- If Space is cold-starting, wait until status is Running before demos
- Keep a backup screen recording of a successful local `python app.py` run if ZeroGPU is exhausted
