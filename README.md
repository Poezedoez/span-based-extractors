# SpERT (original): Span-based Entity and Relation Transformer
# SpEER: Span-based Encoder for Entities and Relations
# SpET: Span-based Entity Transformer
# SpRT: Span-based Relation Transformer

PyTorch code for Span-based extractors. Built on SpERT framework https://github.com/markus-eberts/spert

Check out their paper: "Span-based Entity and Relation Transformer" https://arxiv.org/abs/1909.07755 (accepted at ECAI 2020).

![alt text](assets/Span_extractors.svg)

## Setup
### Requirements
- Required
  - Python 3.5+
  - PyTorch 1.1.0+ (tested with version 1.3.1)
  - transformers 2.2.0+ (tested with version 2.2.0)
  - scikit-learn (tested with version 0.21.3)
  - tqdm (tested with version 4.19.5)
  - numpy (tested with version 1.17.4)
- Optional
  - jinja2 (tested with version 2.10.3) - if installed, used to export relation extraction examples
  - tensorboardX (tested with version 1.6) - if installed, used to save training process to tensorboard

### Fetch data
Fetch converted (to specific JSON format) CoNLL04 \[1\] (we use the same split as \[4\]), SciERC \[2\] and ADE \[3\] datasets (see referenced papers for the original datasets):
```
bash ./scripts/fetch_datasets.sh
```

Fetch model checkpoints (best out of 5 runs for each dataset):
```
bash ./scripts/fetch_models.sh
```
The attached ADE model was trained on split "1" ("ade_split_1_train.json" / "ade_split_1_test.json") under "data/datasets/ade".

## Examples

```
