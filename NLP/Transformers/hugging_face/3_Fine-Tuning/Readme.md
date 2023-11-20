# Fine-Tuning a Pre-trained model

## Processing the data

**Datasets library**
The 🤗 Datasets library provides a very simple command to download and cache a dataset on the Hub. By default, the datasets are stored
at this path:- `~/.cache/huggingface/datasets` which can be chnaged by updating `HF_HOME` env variable.

**Preprocessing**
1. We can use the tokenizer to preprocess the whole dataset but it will require RAM to fit all dataset.
2. We can use the Dataset.map() method which works by applying a function on each element of the dataset.
   1. **batched**- The function can be applied to multiple elements at once.
   2. Datasets library applies this processing is by adding new fields or updating the current fields to the datasets
   3. Tokenizers library already uses multiple threads to tokenize our samples faster. We can also use multiprocessing when applying your preprocessing function with map() by passing along a num_proc argument.

### Dynamic Padding
To apply padding as necessary on each batch and avoid having over-long inputs with a lot of padding (using the max length sentence in the dataset).

**Collate Function**- Responsible for putting together samples inside a batch. 

## Fine-tuning with the trainer API

The 🤗 Transformers provides a **Trainer** class to fine-tune any of the pre-trained models. Before using Trainer, we need to define instance a **TrainingArguments** class 
that will contain all the hyperparameters as well as the checkpoints directory.

**Evaluation**- We can define our own **compute_metrics** function and pass it in the trainer as argument.

