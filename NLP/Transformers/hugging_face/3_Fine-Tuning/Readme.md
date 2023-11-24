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

## Fine-tuning/Training

The 🤗 Transformers provides a **Trainer** class to fine-tune any of the pre-trained models. Before using Trainer, we need to define instance a **TrainingArguments** class 
that will contain all the hyperparameters as well as the checkpoints directory. For evaluation, we can define our own **compute_metrics** function and pass it in the trainer as argument. A simple fine-tuning example can be seen in the notebook - **Fine_tuning_a_model_with_the_Trainer_API_or_Keras.ipynb**.

Trainer class perform multiple steps under the hood, the major ones can be seen in the notebook- **A_full_training.ipynb**. The major steps for training after data processing are as follows:-
1. Apply postprocessing on the **tokenized** datasets and define dataloaders for training and validation set.
2. Define the optimizer and the rate scheduler.
3. Define training and evaluation loop, along with proper device placement of the tensors.

#### Accelerate
The training loop we defined earlier works fine on a single CPU or GPU. But using the 🤗 Accelerate library, with just a few adjustments we can enable distributed training on multiple GPUs or TPUs. A simple training example can be seen in the **A_full_training.ipynb** notebook. 
1. Create instance of **Accelerator**
2. Pass the model, data loaders and optimizer to **.prepare()** method, no need for manual placement after that.
3. Use **Accelerator** instance methods like **backward()** instead of **loss.backward()**



