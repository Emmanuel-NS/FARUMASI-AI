# Demo video — full read-aloud script

**Time:** about 7 to 8 minutes  
**Before you press record:**
1. Open this page and stay signed in: https://huggingface.co/spaces/Emmanuel-NS/farumasi-pharmacist-qa  
2. Wait until the Space says **Running** (first load can take 1–2 minutes).  
3. Also open GitHub in another tab: https://github.com/Emmanuel-NS/FARUMASI-AI  
4. Have the PDF open if you want: `report/Emmanuel_Nsabagasani_FARUMASI_Pharmacist_QA_Report.pdf`  
5. Speak slowly. When you see `[SHOW: ...]`, do that action, then keep reading.

Copy the spoken lines below into a teleprompter app, or print this file and read it.

---

## PART 1 — Introduction (about 30 seconds)

**Say:**

Hello. My name is Emmanuel Nsabagasani. This video is my demo for the Natural Language Processing summative project.

The project is called FARUMASI Pharmacist Assistant. It is a domain-specific English question-answering system for medicine and minor-ailment questions in Rwanda.

In this video I will explain the problem, the data, the models I trained, the experiments, the evaluation, and then I will demonstrate the live web application.

---

## PART 2 — Problem and why I built this (about 1 minute)

**Say:**

The idea came from FARUMASI, a digital pharmacy platform I have been building in Rwanda.

While working with pharmacies, I noticed that patients ask the same basic questions again and again. What is this medicine for? What are the side effects? How should I take it? Pharmacists can answer these questions, but they are busy. They also have to dispense medicines, check prescriptions, and counsel people.

So the goal of this project is to answer routine informational questions from trusted leaflet text, and to send people to a pharmacist when the answer is missing or the situation is risky. That can reduce the number of repeated questions directed at busy pharmacies.

I did not use a commercial chatbot API. ChatGPT and similar tools can invent doses. My system only answers from a fixed corpus of trusted passages, and it must refuse when it does not know.

The language is English only for this version. Most official labels and training resources are in English. Patients in Rwanda often still use English brand names, so I also map local phrases such as panadol to paracetamol, and after sex pills to levonorgestrel.

Target users are adult patients and caregivers. This is not a doctor and not a pharmacist. Every answer has a disclaimer. Emergencies go to SAMU on nine one two.

---

## PART 3 — Dataset (about 45 seconds)

`[SHOW: GitHub README — scroll to the Data table]`

**Say:**

Here is the dataset summary in my GitHub README.

I used human-written sources only. No generated gold answers.

First, openFDA drug labels for about one hundred and forty WHO essential medicines.  
Second, MedQuAD for condition and health-topic questions.  
Third, MedicationQA for real consumer medicine questions.  
Fourth, Rwanda FDA patient information leaflets for local brands such as Coartem.  
Fifth, CLINC150 for out-of-domain negatives.  
And a small challenge set of fifty-six informal questions that I labelled myself for evaluation.

In total there are eight thousand eight hundred and forty passages, and about nine thousand five hundred question-answer pairs. Splits are grouped by medicine so the same drug does not leak from train into test.

A limitation is that some openFDA questions are templated, and many Rwanda PDFs were scanned images with no text layer, so those were excluded.

---

## PART 4 — How the model works (about 2 minutes) — READ CAREFULLY

`[SHOW: GitHub README — System overview diagram, or open pharmqa/pipeline.py briefly]`

**Say:**

Now I will explain the model pipeline, step by step.

When a user types a question, it first goes through simple rules. Greetings get a short hello. Feedback is recorded. A risk detector looks for emergencies such as overdose or chest pain. If it is an emergency, the system tells the person to seek urgent care and call nine one two.

Next is the intent classifier. I fine-tuned DistilBERT, which is a small Transformer encoder. The input is the tokenised question, maximum length sixty-four. The output is one of three labels: medicine information, condition advice, or out of domain. If it is out of domain, we refuse and we never retrieve medicine text.

Then I expand the query. Local lay terms and brand names are rewritten into forms the corpus understands. For example panadol becomes paracetamol. Coartem stays linked to artemether and lumefantrine. After sex pills becomes levonorgestrel.

Then comes the retriever. I fine-tuned all-MiniLM-L6-v2, a bi-encoder. It turns the question and each passage into a three hundred and eighty-four dimensional embedding. Ranking is by cosine similarity. I also boost passages when the medicine name matches, and when the section matches what was asked, for example side effects versus dosage.

Finally comes the answer reader. The deployed model is Qwen two point five, one point five billion, Instruct, adapted with LoRA. LoRA means Low-Rank Adaptation. I only train small adapter matrices instead of the full model, which fits on a free Colab T4 GPU.

Important hyperparameters for the Qwen LoRA run: LoRA rank sixteen, alpha thirty-two, one epoch, batch size two with gradient accumulation eight, learning rate two times ten to the minus four, maximum length one thousand five hundred and thirty-six. Loss is computed only on the answer tokens, not on the prompt. I also trained a Flan-T5 base LoRA reader as a fallback.

The reader must stay grounded. It only sees retrieved on-topic passages. If the product is not in the sources, it should say so instead of inventing ingredients.

The output shown to the user is the answer, the source passages, and a disclaimer.

I trained the chat reader using the Colab notebook and the script under pharmqa generation train chat lora.

---

## PART 5 — Baselines and experiments (about 1 minute 15 seconds)

`[SHOW: README Results tables, or the PDF Tables 3 to 5]`

**Say:**

I did not only train one model. I set baselines and ran controlled experiments.

For intent, the baseline was TF-IDF with logistic regression. Then DistilBERT. The key experiment was training with versus without real patient questions. Models looked almost perfect on the formal test set, about ninety-nine percent accuracy, but they failed on informal consumer questions until I added patient-style data. Challenge accuracy rose to about eighty percent with DistilBERT plus patient questions. The lesson is that formal test scores can hide style overfitting.

For retrieval, the baselines were BM25 and zero-shot MiniLM. Then I fine-tuned MiniLM. Recall at one went from about point three two for BM25 to about point seven zero after fine-tuning. I also tested training with and without Rwanda leaflet questions. Adding Rwanda data helped only a little on local brands. The main gain came from learning question-to-section matching.

For generation, baselines were extractive answers, Flan-T5 zero-shot, and Qwen zero-shot. Proposed models were Flan-T5 with LoRA and Qwen with LoRA. I evaluated with gold context, to isolate the reader, and with retrieved context, for end-to-end behaviour.

The main result: with gold context, Qwen LoRA reached ROUGE-L of point six four three, versus point three two seven for zero-shot Qwen. Passage support rose from about point six two to point nine six. Abstention on unrelated passages rose from twenty-five percent to seventy-three percent. So LoRA taught the model to stay grounded and to refuse more often when the sources do not help.

---

## PART 6 — Metrics and errors (about 45 seconds)

**Say:**

I chose metrics that match each component. Accuracy and macro F1 for intent. Recall at k and MRR for retrieval. ROUGE-L and BERTScore for answer wording. Support measures how many answer content words appear in the passages. Abstention measures honest “I do not know” behaviour.

These metrics are not clinical correctness. They measure agreement with trusted text and scope control.

I will now show successes and failures live on the web app.

---

## PART 7 — Live web demo (about 2 minutes 30 seconds)

`[SHOW: Hugging Face Space chat interface. Paste each question, wait for the answer, then speak.]`

**Say before the first question:**

This is the deployed Gradio web application on Hugging Face Spaces. Behind the interface, the app calls my PharmacistAssistant answer method. DistilBERT, MiniLM, and Qwen with LoRA run on ZeroGPU for each reply. The user sees the chat answer and the source passages.

### Demo 1 — greeting

`[TYPE: hey]`

**Say:**

This is a greeting. The rules handle it without calling the full reader. The system introduces itself briefly.

### Demo 2 — success: metformin side effects

`[TYPE: What are the side effects of metformin?]`

**Say while it answers:**

This is a successful case. The intent is medicine information. The retriever finds metformin passages. The Qwen LoRA reader opens with a side-effects framing and lists effects from the trusted label, such as diarrhoea or nausea. Below the answer you can see the source passages.

`[TYPE: and how should I take it?]`

**Say:**

This is a follow-up. The system keeps metformin as the topic and answers about how to take it, still from the retrieved text.

### Demo 3 — success: emergency contraception lay language

`[TYPE: after sex pills]`

**Say:**

Here the query expansion maps after sex pills to levonorgestrel. The system gives an overview from the leaflet text.

`[TYPE: are there any of 120hrs]`

**Say:**

Now it should discuss the one hundred and twenty hour option, ulipristal, still grounded in sources. This shows local patient phrasing, not only formal drug names.

### Demo 4 — fuzzy name: quarteme

`[TYPE: what do you know about quarteme]`

**Say:**

Quarteme is a misspelling or near name for Coartem. The system should suggest Coartem and give a malaria overview for artemether and lumefantrine, instead of inventing a fake product.

### Demo 5 — honest failure: unknown product

`[TYPE: what do you know about kopavitamins]`

**Say:**

This is an unsuccessful or honest-refusal case. Kopavitamin is not in my leaflet corpus. A good system must not invent ingredients. Mine should say the product is not in the sources, and suggest asking a pharmacist. That behaviour is important for FARUMASI, because a wrong confident answer is worse than no answer.

### Demo 6 — out of scope

`[TYPE: what is the exchange rate of the rwandan franc?]`

**Say:**

This is out of domain. The DistilBERT intent classifier should refuse. No medicine answer is invented.

---

## PART 8 — GitHub and reproducibility (about 30 seconds)

`[SHOW: GitHub repo — README reproduce section, then open the results folder]`

**Say:**

Everything needed to reproduce the project is on GitHub under Emmanuel-NS FARUMASI-AI.

The README has setup instructions. The notebooks folder has the Colab training notebooks. The results folder has the JSON metric files that match the tables in my report. Anyone can clone the repo, prepare the data, and run the app locally with python app.py.

---

## PART 9 — Closing / ownership (about 30 seconds)

**Say:**

To close: I defined the problem from FARUMASI and from busy community pharmacies. I prepared the datasets, fine-tuned DistilBERT, MiniLM, and the Qwen LoRA reader, ran the baselines and experiments, analysed errors, and deployed the Gradio app.

AI coding tools helped me scaffold some code, but I ran the training, checked the metrics, tested the live app, and I take full responsibility for this submission.

Limitations: English only, not clinical advice, and some local brands are missing from the leaflets. Future work is multilingual support, clinician review, and tighter integration into FARUMASI so routine questions are answered before they reach a busy pharmacy counter.

Thank you for watching. Links to the GitHub repository, the live Space, and the research report PDF are in the submission.

---

## Quick cheat sheet (if you freeze)

| Topic | One sentence |
|---|---|
| Problem | Answer routine medicine questions from trusted text so busy pharmacies get fewer repeated questions. |
| Models | DistilBERT intent, MiniLM retriever, Qwen 1.5B + LoRA reader. |
| Best number | Qwen LoRA ROUGE-L 0.643 vs zero-shot 0.327; support 0.96 vs 0.62. |
| Why LoRA | Train small adapters on free Colab without updating the full model. |
| Demo success | Metformin side effects; after sex pills; quarteme to Coartem. |
| Demo fail | Kopavitamins not in sources; franc exchange rate out of scope. |
| Deployment | Gradio on Hugging Face Spaces, ZeroGPU for each answer. |

## Questions they might ask you (short answers)

**Why not just use ChatGPT?**  
Because it can invent doses, and the brief forbids a plain API chatbot. My answers must come from a known corpus.

**What is LoRA?**  
Low-Rank Adaptation. I freeze the base Qwen weights and train small rank-sixteen matrices. That is how I fine-tuned on a free T4.

**What is support?**  
The share of answer content words that also appear in the retrieved passages. Higher support means more grounded answers.

**Why English?**  
Training labels and leaflets with usable text are mostly English. Multilingual is future work.

**Is this medical advice?**  
No. It is an informational prototype with a disclaimer. Emergencies go to SAMU 912.
