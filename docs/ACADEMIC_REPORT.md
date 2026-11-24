# Fine-Tuning BERT for SST-2 Sentiment Classification

## Abstract

This project investigates transfer learning for natural-language sentiment classification by fine-tuning `bert-base-uncased` on the Stanford Sentiment Treebank binary classification task, SST-2. The work demonstrates dataset preprocessing, WordPiece tokenization, dynamic padding, supervised fine-tuning, validation with standard classification metrics, and visualization of multi-head self-attention. A saved historical checkpoint reports validation accuracy of `0.8700` and F1 score of `0.8746`; however, its recorded run configuration does not match the earlier notebook version, so those values are treated as historical observations rather than final reproducible results. The revised experiment pins external data and model revisions, fixes seeds and batch size, isolates run artifacts, and documents the limitations of interpreting attention as explanation.

## 1. Introduction

Sentiment analysis assigns an opinion polarity to text. Traditional approaches often require task-specific feature engineering, while pretrained Transformer models learn contextual language representations from large corpora and can be adapted to downstream tasks using comparatively small labeled datasets.

BERT uses bidirectional self-attention to construct contextual token representations. For sequence classification, the final representation associated with the special `[CLS]` token is passed to a classification head. This project studies both the predictive performance of fine-tuned BERT and the attention patterns produced for a contrastive sentence containing positive and negative clauses.

## 2. Objectives

1. Load and preprocess the SST-2 binary sentiment dataset.
2. Fine-tune a pretrained BERT model for sequence classification.
3. Evaluate performance using accuracy, precision, recall, and F1 score.
4. Extract attention tensors from all Transformer layers during inference.
5. Visualize token-to-token and `[CLS]` attention distributions.
6. Critically discuss reproducibility and the limitations of attention-based interpretation.

## 3. Dataset

The experiment uses the SST-2 configuration of the GLUE benchmark. Each labeled example contains a sentence and a binary sentiment label:

- `0`: negative sentiment
- `1`: positive sentiment

The corrected project loads `nyu-mll/glue` at immutable revision `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c`. The official training and validation splits are preserved. Test labels are not used because the public GLUE test split does not expose ground-truth labels for local evaluation.

For an explicitly requested smoke test, the shuffled training split is reduced to 500 examples and validation to 100 examples. Full training is the default on both CPU and GPU.

## 4. Methodology

### 4.1 Tokenization

The project uses the tokenizer distributed with `google-bert/bert-base-uncased` at revision `86b5e0934494bd15c9632b12f734a8a67f723594`. Sentences are converted into WordPiece token IDs and truncated to 64 tokens. Padding is applied dynamically to the longest sequence in each batch rather than statically padding the entire dataset.

### 4.2 Model architecture

The base model contains 12 Transformer encoder layers, 12 attention heads per layer, and a hidden size of 768. A two-class sequence-classification head is added above the pooled `[CLS]` representation. The saved configuration maps class IDs to the human-readable labels `Negative` and `Positive`.

### 4.3 Fine-tuning configuration

The corrected default experiment uses:

- random seed: 42
- epochs: 2
- batch size: 16
- evaluation batch size: 32
- learning rate: `2 × 10⁻⁵`
- weight decay: `0.01`
- warmup ratio: `0.10`
- maximum token length: 64
- best-model criterion: validation F1
- checkpoint retention: two checkpoints

Attention output is disabled during training and validation because collecting every attention tensor increases memory consumption without contributing to the classification loss. It is enabled only for inference and visualization.

### 4.4 Evaluation

The validation predictions are evaluated with accuracy, binary precision, binary recall, and binary F1 score. F1 is used for selecting the best checkpoint. The corrected pipeline saves evaluation metrics, Trainer state, and run metadata alongside the resulting model.

## 5. Historical results

The supplied historical checkpoint records:

- validation accuracy: `0.8700`
- validation precision: `0.8718`
- validation recall: `0.8774`
- validation F1: `0.8746`
- validation loss: `0.4368`

These metrics came from a 125-step run evaluated on approximately 300 examples. The earlier notebook visible in the workspace specified 500 training and 100 validation examples for its CPU path, which would produce a different number of training steps. Therefore, the historical values demonstrate that the saved model learned the task, but they must not be presented as the final result of the corrected experiment.

A valid final academic result requires rerunning the corrected notebook or CLI and reporting the newly generated metric file together with `run_metadata.json`.

## 6. Attention visualization

After training, the model is reloaded using the eager attention implementation. For an input sentence, the project extracts one attention tensor per encoder layer. Each tensor has dimensions:

`batch size × attention heads × sequence length × sequence length`

Two static visualizations are provided:

1. A token-to-token heatmap for a selected layer and head.
2. A bar chart of final-layer attention originating from `[CLS]`, averaged across heads.

The example sentence—“The movie was absolutely fantastic, but the acting was terrible.”—contains contrastive sentiment. It is useful for observing how different heads distribute attention across positive vocabulary, negative vocabulary, punctuation, special tokens, and the conjunction “but.”

## 7. Discussion

The historical validation metrics are plausible for a BERT model adapted to SST-2, showing that pretrained contextual representations transfer effectively to binary sentiment classification. Dynamic padding reduces unnecessary computation, while disabling attention tensors during training avoids large intermediate outputs.

Attention heatmaps can reveal local, long-range, delimiter, and special-token patterns. Nevertheless, a large attention value does not establish that a token caused the classifier's decision. Attention weights should be treated as diagnostic observations and, where explanation is required, complemented by perturbation or attribution methods.

Softmax outputs are reported as predicted probabilities. They are not described as confidence scores because the model has not undergone probability calibration.

## 8. Reproducibility and ethical considerations

The project pins the external dataset and base-model revisions, fixes random seeds, uses identical default sample counts and batch size across hardware, and records run metadata. Exact numerical equality may still vary across PyTorch versions, CUDA kernels, drivers, and hardware.

SST-2 contains movie-review language and is not representative of every domain, language variety, or demographic group. A classifier trained on this dataset should not be generalized to high-stakes decisions or treated as a universal detector of human sentiment.

## 9. Limitations

- The final corrected experiment has not yet been executed in the current environment.
- Historical metrics do not match the earlier notebook's recorded sampling configuration.
- The maximum length of 64 tokens is tailored to short SST-2 sentences.
- Only a single pretrained architecture and primary hyperparameter setting are studied.
- Attention visualization does not provide causal explanation.
- No probability-calibration or subgroup-bias analysis is included.

## 10. Future work

- Rerun the corrected full experiment and report the generated metrics.
- Compare BERT with a recurrent baseline and a simpler bag-of-words classifier.
- Conduct learning-rate, batch-size, and maximum-length ablations.
- Evaluate multiple random seeds and report mean and standard deviation.
- Add a confusion matrix and qualitative error analysis.
- Compare attention with gradient-based or perturbation-based attribution.
- Evaluate calibration using reliability diagrams and expected calibration error.

## 11. Conclusion

The project demonstrates an end-to-end academic deep-learning workflow for fine-tuning BERT on SST-2 and inspecting multi-head self-attention. The revised implementation improves experimental validity without changing the educational focus. Its primary contribution is a transparent, reproducible demonstration of transfer learning and attention visualization, accompanied by an explicit account of what the historical metrics can and cannot establish.

## References

1. Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.” NAACL-HLT, 2019.
2. Wang, A. et al. “GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding.” ICLR, 2019.
3. Socher, R. et al. “Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank.” EMNLP, 2013.
4. Vaswani, A. et al. “Attention Is All You Need.” NeurIPS, 2017.
5. Clark, K. et al. “What Does BERT Look At? An Analysis of BERT's Attention.” BlackboxNLP, 2019.
6. Jain, S. and Wallace, B. C. “Attention is not Explanation.” NAACL-HLT, 2019.

