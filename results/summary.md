# Experiment summary

## Dataset

- Corpus passages: 8250 ({'openFDA': 2049, 'MedQuAD': 5546, 'MedicationQA': 655})
- QA pairs: 8758 ({'MedQuAD': 5580, 'openFDA-templated': 2523, 'MedicationQA': 655})
- train: 7028 questions, 1652 distinct medicines/conditions
- val: 803 questions, 207 distinct medicines/conditions
- test: 927 questions, 207 distinct medicines/conditions

## Intent classification

| model                       |   test acc |   test macro-F1 |   MedicationQA acc |   challenge acc |   challenge macro-F1 |
|:----------------------------|-----------:|----------------:|-------------------:|----------------:|---------------------:|
| tfidf_lr_word               |     0.9916 |          0.9912 |             0.4722 |          0.4107 |               0.3849 |
| tfidf_lr_word_char          |     0.9943 |          0.994  |             0.6667 |          0.4821 |               0.4486 |
| tfidf_lr_word_char_consumer |     0.9791 |          0.9799 |             0.5556 |          0.7857 |               0.7843 |
| tfidf_lr_word_consumer      |     0.9782 |          0.9791 |             0.5    |          0.7321 |               0.7289 |

## Retrieval (test split)

| model           |   recall@1 |   recall@5 |   recall@10 |   mrr@10 |   topic@1 |
|:----------------|-----------:|-----------:|------------:|---------:|----------:|
| bm25            |     0.2891 |     0.6332 |      0.794  |   0.4356 |    0.8447 |
| minilm_zeroshot |     0.4347 |     0.7929 |      0.8932 |   0.5815 |  nan      |

### Retrieval by data source

| model           | source            |   n |   recall@1 |   recall@5 |   mrr@10 |
|:----------------|:------------------|----:|-----------:|-----------:|---------:|
| bm25            | MedQuAD           | 532 |     0.2744 |     0.6598 |   0.4348 |
| bm25            | MedicationQA      |  78 |     0.4744 |     0.7564 |   0.5987 |
| bm25            | openFDA-templated | 317 |     0.2681 |     0.5584 |   0.3967 |
| minilm_zeroshot | MedQuAD           | 532 |     0.4492 |     0.8008 |   0.5921 |
| minilm_zeroshot | MedicationQA      |  78 |     0.6667 |     0.8846 |   0.7558 |
| minilm_zeroshot | openFDA-templated | 317 |     0.3533 |     0.7571 |   0.5208 |

## Answer generation

| system     | context   |   n |   ROUGE-L |   BERTScore-F1 |   avg words |
|:-----------|:----------|----:|----------:|---------------:|------------:|
| extractive | retrieved |  60 |     0.344 |         0.8098 |        64.2 |

## Safety risk detector (challenge set)

Accuracy: 1.0

|               |   precision |   recall |   f1-score |   support |
|:--------------|------------:|---------:|-----------:|----------:|
| routine       |           1 |        1 |          1 |        34 |
| prescription  |           1 |        1 |          1 |         4 |
| special_group |           1 |        1 |          1 |         9 |
| emergency     |           1 |        1 |          1 |         9 |
