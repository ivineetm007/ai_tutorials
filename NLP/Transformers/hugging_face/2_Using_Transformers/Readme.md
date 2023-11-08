# Using Transformers
Transformer models are usually very large with millions to tens of billions of parameters; training and deploying these models is a complicated undertaking. Furthermore, with new models being released on a near-daily basis, trying them all out is no easy task.

The 🤗 Transformers library was created to solve this problem. Its goal is to provide a single API through which any Transformer model can be loaded, trained, and saved. The library’s main features are:

1. **Ease of use**: Downloading, loading, and using a state-of-the-art NLP model for inference can be done in just two lines of code.
2. **Flexibility**: At their core, all models are simple PyTorch nn.Module or TensorFlow tf.keras.Model classes and can be handled like any other models in their respective machine learning (ML) frameworks.
3. **Simplicity**: Hardly any abstractions are made across the library. The “All in one file” is a core concept: a model’s forward pass is entirely defined in a single file. Unlike other ML frameworks, the models are not built on modules that are shared across files; instead, each model has its own layers. 

## Behind the pipeline
**Behind_the_pipeline_(PyTorch.ipynb)** - The notebook provides a complete example of using the HuggingFace library for sentiment analysis. It demonstrates the three steps of the pipeline: preprocessing, passing inputs through the model, and postprocessing.
<p align="center"><img src="img.png" width="400" align="center"/></p>

1. **Tokenizer**- It is responsible for splitting the input into words, subwords, or symbols (like punctuation) that are called tokens, mapping each token to an integer and adding additional inputs like padding. Use **AutoTokenizer** class to load any tokenizer given model or pipeline. 
2. **Model**- It contains the base Transformer module which give **hidden states** as output. Use **AutoModel** to load the base model.
3. **Head**- The model heads take the high-dimensional vector of hidden states as input and project them onto a different dimension. There are differnt classes based on the task like **AutoModelForSequenceClassification** for sequence classification head.
4. **Post Processing**- Conversion of raw logits into classes, scores etc.  
<p align="center"><img src="img_1.png" width="600" align="center"/></p>

## Models
1. Loading Model
   1. Use **AutoModel** that automatically guess the appropriate model architecture for your checkpoint
   2. If known, use the Model and it's config class directly like 
   ```
   from transformers import BertConfig, BertModel

   # Building the config
   config = BertConfig()

   # Building the model from the config
   model = BertModel(config)# Randomly initialized
   ```
   3. Use the **from_pretrained** method either with known model class or AutoModel. The weights have been downloaded and cached in the cache folder, which defaults to `~/.cache/huggingface/transformers`; change it by setting the `HF_HOME` environment variable.
2. Saving Model- Use the **save_pretrained(<PATH>)** method. It saves two file for the model
   1. `config.json`- Attributes necessary to build the model architecture
   2. `pytorch_model.bin`- It contains all the model's weights.

## Tokenizer
It translates raw text into numerical data processed by model. major tokenization types
1. `Word-based`- Split raw text into words. One simple way is to use whitespace to tokenize the text into words. Drawbacks- It leds to pretty **large vocabularies**.
2. `Character-based` - Split the text into characters, rather than words. It has some benefits- **small vocabulary** and **few out-of-vocabulary** tokens. Drawbacks- Large number of tokens to be processed by model 
3. `Subword tokenization` - Frequently used words should not be split into smaller subwords, but rare words should be decomposed into meaningful subwords. Ex. “annoyingly” will be decomposed into "annoying" + "ly" 
4. `Other Advanced techniques` - Byte-level BPE (GPT-2), WordPiece (BERT), SentencePiece

**Loading and Saving**- It’s based on the same two methods: `from_pretrained()` and `save_pretrained()`.
Tokenizer Methods
1. `Encoding`- Translating input text to numbers in two-step process:- 
   1. Tokenization- Split the words until it obtains tokens that can be represented by its vocabulary.
   2. Token to input ID- Convert tokens to vocabulary IDs.
2. `D3coding` - From vocabulary indices to the final string.

### Multiple Sentences
**Batching** is the act of sending multiple sentences through the model, all at once.  Since, the model accepts tensors of rectangle shape. We need to make all the sentences have equal lengths
1. `Padding` makes sure all our sentences have the same length by adding a special word called the padding token to the sentences with fewer values.
2. `Attention masks` are tensors with the exact same shape as the input IDs tensor, filled with 0s and 1s: 1s indicate the corresponding tokens should be attended to.
3. `Longer sequences`- Most models handle sequences of up to 512 or 1024 tokens. For longer sequences- Use a model with a longer sequence length or truncate sequences
### Tokenizer call
It handles the batching, padding and long sequences by default.
1. Padding
```
 Will pad the sequences up to the maximum sequence length
model_inputs = tokenizer(sequences, padding="longest")

# Will pad the sequences up to the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, padding="max_length")

# Will pad the sequences up to the specified max length
model_inputs = tokenizer(sequences, padding="max_length", max_length=8)
```
2. Truncation
```
 Will pad the sequences up to the maximum sequence length
model_inputs = tokenizer(sequences, padding="longest")

# Will pad the sequences up to the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, padding="max_length")

# Will pad the sequences up to the specified max length
model_inputs = tokenizer(sequences, padding="max_length", max_length=8)
```

**Special Tokens**
If we take a look at the input IDs returned by the tokenizer, we will see they are a tiny bit different from what we had earlier:
```
sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
print(model_inputs["input_ids"])

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
>>> [101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012, 102]
>>> [1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012]

# One token ID was added at the beginning, and one at the end. Let’s decode the two sequences of IDs above to see what this is about:
print(tokenizer.decode(model_inputs["input_ids"]))
print(tokenizer.decode(ids))
>>> "[CLS] i've been waiting for a huggingface course my whole life. [SEP]"
>>> "i've been waiting for a huggingface course my whole life."

```
The tokenizer added the special word [CLS] at the beginning and the special word [SEP] at the end. Note that some models don’t add special words, or add different ones; models may also add these special words only at the beginning, or only at the end.