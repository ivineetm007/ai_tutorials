

### Transformer model
**Transformers,_what_can_they_do_.ipynb**- It provides an overview of transformers and their widespread use in various NLP tasks. It introduces the concept of **pipelines** in the Hugging Face Transformers library, which connect models with preprocessing and postprocessing steps. It explains how pipelines can be used for tasks such as **sentiment analysis, zero-shot classification, text generation, mask filling, named entity recognition, question answering, summarization, and translation**. Examples and code snippets are provided to demonstrate the usage of these pipelines. It also highlights the availability of pretrained models in the **Hugging Face Model Hub** and the option to upload custom models. It concludes by mentioning the **Inference API**.

Broadly, transformer models can be grouped into three categories:-
1. GPT-like (also called auto-regressive Transformer models)
2. BERT-like (also called auto-encoding Transformer models)
3. BART/T5-like (also called sequence-to-sequence Transformer models)

**Bias and limitations**

 To enable pretraining on large amounts of data, researchers often scrape all the content they can find, taking the **best as well as the worst of what is available on the internet**. Below is one example where the model gives only one gender-free answer (waiter/waitress).:-
```
from transformers import pipeline

unmasker = pipeline("fill-mask", model="bert-base-uncased")
result = unmasker("This man works as a [MASK].")
print([r["token_str"] for r in result])

result = unmasker("This woman works as a [MASK].")
print([r["token_str"] for r in result])
```
```
['lawyer', 'carpenter', 'doctor', 'waiter', 'mechanic']
['nurse', 'waitress', 'teacher', 'maid', 'prostitute']
```
When you use these tools, you therefore need to keep in the back of your mind that the original model you are using could very easily generate sexist, racist, or homophobic content. Fine-tuning the model on your data won’t make this intrinsic bias disappear. A less obvious source of bias is the way the model is trained. Your model will blindly optimize for whatever metric you chose, without any second thoughts.

### Using Transformers

**Behind_the_pipeline_(PyTorch.ipynb)** - The notebook provides a complete example of using the HuggingFace library for sentiment analysis. It demonstrates the three steps of the pipeline: preprocessing, passing inputs through the model, and postprocessing.

# References
1. All the notebooks are from hugging face NLP course- https://huggingface.co/learn/nlp-course/chapter1/1