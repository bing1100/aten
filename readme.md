# Medical Classification and NER

Solutions are split into two separate jupyter notebooks. Both require the following libraries:

```
pip install -r requirements.txt
```

Each notebook is a standalone solution. Training, evaluation, and inference workflows are contained in order in each notebook. 

We provide a brief description of each approach in the following, with extended descriptions self-contained in each notebook. 

## Classification with HF

We achieve 100% performance on metrics of accuracy, precision, recall, and F1 over a testset of the provided classification data. All code is in ```1. classification hf.ipynb```.

### Methods

We utilize and fine-tune a ```distilbert-base-uncased``` pre-trained model. Although other models can be utilized and may perform better, this is beyond the scope of our current work as we cannot measure performance above what is already achieved with the provided dataset. 

Hyperparameter tuning is utilized over 5 trials to find best performing configuration:

```python
def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float(...),
        "per_device_train_batch_size": trial.suggest_categorical(...),
        "num_train_epochs": trial.suggest_categorical(...),
        "weight_decay": trial.suggest_float(...)
    }
```

The best performing configuration is then utilized for fine-tuning with the model saved for inference-only use. Inference-only workflow is provided in the standalone notebook

#### Data Processing
Classification data is sub-divided into train, validation, and test splits using a 70/10/20 split.

### Improvements and Production

- Data augmentation: synthetic data from LLMs can be generated to supplement the provided classification data
    - Incorporating in spelling mistakes, other symptoms, alternative sentence structures, etc.
- Smaller models \w ensemble learning can be utilized to further reduce computational footprint and is better suited for production

Fine-tuned model can be deployed in kubernetes for batch processing of large amount of input messages and texts. 

## NER with SpaCy

We achieve 100% performance on metrics of precision, recall, and F1 over provided NER data. All code is in ```2. ner spacy.ipynb```.

### Methods
We utilized two specialized pre-trained bio/medical SpaCY models collected from HF, avoiding training of new NER model.

```en_core_med7_lg``` is specialized for NER on drug doses.

```en_ner_bc5cdr_md``` is specialized for NER on symptoms.

We build a logical model for NER on drug names, allowing for easy addition and removal of drugs depending on country and over time. This logical model relies on a dataset of all FDA drugs between 1939-Present collected from https://www.kaggle.com/datasets/protobioengineering/united-states-fda-drugs-feb-2024?resource=download. 

We define batch processing for NER for online inference. We test examples OOD to the original data and show generalizability of our approach. 

### Improvements and Production
- Data augmentation: additional symptoms, drugs, and possible dosage can be integrated to expand the dataset. Existing pre-trained models can be used to label additional data to add to the dataset.
- SpaCy model training can be used to build a better specialized model capable of solving the task directly - this is beyond the scope of our current work as we cannot measure performance above what is already achieved with the provided dataset.

Fine-tuned model can be deployed in kubernetes for batch processing of large amount of input messages and texts. 

