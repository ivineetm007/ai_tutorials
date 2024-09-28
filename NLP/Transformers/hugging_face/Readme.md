
1. **Transformers Models**- In this chapter, we learn how to use the `pipeline()` function to solve NLP tasks, overview of the Transformer architecture and types of models- Encoder, Decoder and Encoder-Decoder architectures.
2. **Using Transformers**- This chapter starts with an end-to-end example that demonstrates using a model and tokenizer to 
 replicate the pipeline() function from Chapter 1. It explores the model API, discussing loading models and processing numerical inputs for predictions. The tokenizer API is also covered, which handles text-to-numerical (vice-versa) conversion for neural networks. The chapter concludes by explaining how to process multiple sentences in a prepared batch and provides an overview of the high-level **tokenizer()** function.
3. **Fine Tuning**- This chapter covers the data preparation and the training of the models. It covers the downloading and processing of the datasets from the hub. For training, it covers the high level **Trainer API** and also describes the major steps under the hood through custom training example. In the end, it covers how to leverage the 🤗 **Accelerate** library to easily run that custom training loop on any distributed setup    
4. **Sharing Model and Tokenizers**- 





# References
1. All the notebooks are from hugging face NLP course- https://huggingface.co/learn/nlp-course/chapter1/1