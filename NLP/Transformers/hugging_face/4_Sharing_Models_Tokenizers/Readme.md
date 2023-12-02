# Hugging Face Hub (Sharing Models and Tokenizer)
[Hugging Face](https://huggingface.co/) is a central platform that enables anyone to discover, use, and contribute new state-of-the-art models and datasets.
1. Each of these models is hosted as a Git repository, which allows versioning and reproducibility.
2. Sharing a model on the Hub automatically deploys a hosted Inference API for that model.

## Using Pre-trained model
Three ways to load a pre-trained model
1. `pipeline`- Using pipeline The only thing you need to watch out for is that the chosen checkpoint is suitable for the task it’s going to be used for.
```
from transformers import pipeline
camembert_fill_mask = pipeline("fill-mask", model="camembert-base")
results = camembert_fill_mask("Le camembert est <mask> :)")
```
2. Instantiate the checkpoint using the model architecture directly
```
from transformers import CamembertTokenizer, CamembertForMaskedLM

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertForMaskedLM.from_pretrained("camembert-base")
```
3. Using the **Auto*** classes
```
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForMaskedLM.from_pretrained("camembert-base")
```
## Sharing Pre-trained model

Three ways to create model repositories:-
1. Using the **push_to_hub** API
2. Using the huggingface_hub Python library
3. Using the web interface

Once you’ve created a repository, you can upload files via following methods
1. `upload_file`- Via HTTP Post with limitation of 5GB
2. `Repository`- Abstraction around git and git-lfs
3. Using `git`

## Model Card
The model card usually starts with a very brief, high-level overview of what the model is for, followed by additional details in the following sections:

1. Model description- This includes the architecture, version, if it was introduced in a paper, if an original implementation is available, the author, and general information about the model. Any copyright should be attributed here. General information about training procedures, parameters, and important disclaimers can also be mentioned in this section.
2. Intended uses & limitations- Here you describe the use cases the model is intended for and document areas that are known to be out of scope for the model.
3. How to use- This section should include some examples of how to use the model.
4. Limitations and bias
5. Variable and metrics- Here you should describe the metrics you use for evaluation, and the different factors you are mesuring.
6. Training data- This part should indicate which dataset(s) the model was trained on.
7. Training procedure- In this section you should describe all the relevant aspects of training that are useful from a reproducibility perspective. 
8. Evaluation results- Finally, provide an indication of how well the model performs on the evaluation dataset.


**Model Card Metadata**

You can add metadata as header in the Model card. The categories a model belongs to are identified according to the metadata you add in the model card header. Example
```
---
language: fr
license: mit
datasets:
- oscar
---
```
This metadata is parsed by the Hugging Face Hub, which then identifies this model as being a French model, with an MIT license, trained on the Oscar dataset.
