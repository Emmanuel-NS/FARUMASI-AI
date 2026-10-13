# Experiment summary

## Dataset

- Corpus passages: 8840 ({'openFDA': 2049, 'MedQuAD': 5546, 'MedicationQA': 655, 'RwandaFDA-PIL': 590})
- QA pairs: 9518 ({'MedQuAD': 5580, 'openFDA-templated': 2523, 'RwandaFDA-PIL': 760, 'MedicationQA': 655})
- train: 7200 questions, 1672 distinct medicines/conditions
- val: 1136 questions, 212 distinct medicines/conditions
- test: 1182 questions, 233 distinct medicines/conditions

## Intent classification

| model                       |   test acc |   test macro-F1 |   MedicationQA acc |   challenge acc |   challenge macro-F1 |
|:----------------------------|-----------:|----------------:|-------------------:|----------------:|---------------------:|
| distilbert                  |     0.9941 |          0.9938 |             0.8889 |          0.8036 |               0.7822 |
| distilbert_original         |     0.998  |          0.9978 |             0.9167 |          0.5536 |               0.486  |
| tfidf_lr_word               |     0.9925 |          0.9921 |             0.5278 |          0.3929 |               0.372  |
| tfidf_lr_word_char          |     0.9947 |          0.9943 |             0.6944 |          0.4643 |               0.4202 |
| tfidf_lr_word_char_consumer |     0.979  |          0.9797 |             0.5556 |          0.7679 |               0.7635 |
| tfidf_lr_word_consumer      |     0.9775 |          0.9784 |             0.5    |          0.7321 |               0.7289 |

## Retrieval (test split)

| model                         |   recall@1 |   recall@5 |   recall@10 |   mrr@10 |   topic@1 |
|:------------------------------|-----------:|-----------:|------------:|---------:|----------:|
| bm25                          |     0.3223 |     0.7056 |      0.8316 |   0.478  |    0.8519 |
| hybrid_rrf                    |     0.5508 |     0.8672 |      0.9433 |   0.6841 |    0.9509 |
| hybrid_rrf_no_rwandafda       |     0.5457 |     0.8663 |      0.9433 |   0.6819 |    0.9535 |
| minilm_finetuned              |     0.6954 |     0.9577 |      0.9873 |   0.8103 |    0.9772 |
| minilm_finetuned_no_rwandafda |     0.6937 |     0.9543 |      0.9915 |   0.8092 |    0.9772 |
| minilm_zeroshot               |     0.4357 |     0.8037 |      0.8976 |   0.5888 |    0.8816 |

### Retrieval by data source

| model                         | source            |   n |   recall@1 |   recall@5 |   mrr@10 |
|:------------------------------|:------------------|----:|-----------:|-----------:|---------:|
| bm25                          | MedQuAD           | 613 |     0.2904 |     0.7194 |   0.4639 |
| bm25                          | MedicationQA      |  67 |     0.4627 |     0.7313 |   0.5715 |
| bm25                          | RwandaFDA-PIL     | 180 |     0.4444 |     0.9167 |   0.6402 |
| bm25                          | openFDA-templated | 322 |     0.2857 |     0.5559 |   0.3946 |
| hybrid_rrf                    | MedQuAD           | 613 |     0.5383 |     0.8646 |   0.6735 |
| hybrid_rrf                    | MedicationQA      |  67 |     0.6418 |     0.8806 |   0.7434 |
| hybrid_rrf                    | RwandaFDA-PIL     | 180 |     0.6889 |     0.9667 |   0.8103 |
| hybrid_rrf                    | openFDA-templated | 322 |     0.4783 |     0.8137 |   0.6212 |
| hybrid_rrf_no_rwandafda       | MedQuAD           | 613 |     0.5334 |     0.8679 |   0.6725 |
| hybrid_rrf_no_rwandafda       | MedicationQA      |  67 |     0.6269 |     0.8657 |   0.7309 |
| hybrid_rrf_no_rwandafda       | RwandaFDA-PIL     | 180 |     0.6722 |     0.9667 |   0.7989 |
| hybrid_rrf_no_rwandafda       | openFDA-templated | 322 |     0.4814 |     0.8075 |   0.6239 |
| minilm_finetuned              | MedQuAD           | 613 |     0.7129 |     0.9347 |   0.8061 |
| minilm_finetuned              | MedicationQA      |  67 |     0.7463 |     0.9254 |   0.8264 |
| minilm_finetuned              | RwandaFDA-PIL     | 180 |     0.6778 |     0.9944 |   0.8218 |
| minilm_finetuned              | openFDA-templated | 322 |     0.6615 |     0.9876 |   0.8085 |
| minilm_finetuned_no_rwandafda | MedQuAD           | 613 |     0.7227 |     0.9331 |   0.8145 |
| minilm_finetuned_no_rwandafda | MedicationQA      |  67 |     0.7164 |     0.9104 |   0.806  |
| minilm_finetuned_no_rwandafda | RwandaFDA-PIL     | 180 |     0.6667 |     0.9944 |   0.8101 |
| minilm_finetuned_no_rwandafda | openFDA-templated | 322 |     0.6491 |     0.9814 |   0.7995 |
| minilm_zeroshot               | MedQuAD           | 613 |     0.4861 |     0.8418 |   0.6337 |
| minilm_zeroshot               | MedicationQA      |  67 |     0.6119 |     0.8209 |   0.7064 |
| minilm_zeroshot               | RwandaFDA-PIL     | 180 |     0.4    |     0.7833 |   0.5531 |
| minilm_zeroshot               | openFDA-templated | 322 |     0.323  |     0.7391 |   0.4988 |

## Answer generation

| system     | context   |   n |   ROUGE-L |   BERTScore-F1 |   avg words |   Rwanda PIL ROUGE-L |
|:-----------|:----------|----:|----------:|---------------:|------------:|---------------------:|
| extractive | gold      | 400 |    0.931  |         0.9852 |        62.3 |               0.7693 |
| extractive | retrieved | 400 |    0.3953 |         0.8328 |        59.9 |               0.4596 |
| lora       | gold      | 400 |    0.6077 |         0.8996 |        36.9 |               0.4787 |
| lora       | retrieved | 400 |    0.4138 |         0.8412 |        34.7 |               0.4289 |
| zeroshot   | gold      | 400 |    0.1911 |         0.761  |         8.7 |               0.1472 |
| zeroshot   | retrieved | 400 |    0.1603 |         0.7477 |         8.9 |               0.1378 |

## Safety risk detector (challenge set)

Accuracy: 1.0

|               |   precision |   recall |   f1-score |   support |
|:--------------|------------:|---------:|-----------:|----------:|
| routine       |           1 |        1 |          1 |        34 |
| prescription  |           1 |        1 |          1 |         4 |
| special_group |           1 |        1 |          1 |         9 |
| emergency     |           1 |        1 |          1 |         9 |
