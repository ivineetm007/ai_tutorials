# Introduction to Diffusion and Diffusers
Diffusion models are a class of **generative models**, in which random noise is gradually refined until an output image emerges.
Training the model consists of these steps:-
1. Load in some images from the training data
2. **Add noise**, in different amounts. Remember, we want the model to do a good job estimating how to 'fix' (denoise) both extremely noisy images and images that are close to perfect.
3. Feed the noisy versions of the inputs into the model
4. Evaluate how well the model does at denoising these inputs
5. Use this information to update the model weights

At **inference** time, we begin with a completely **random input** and repeatedly feed it through the model, updating it each time by a small amount based on the model prediction. There are a number of sampling methods that try to streamline this process so that we can generate good images with as few steps as possible.

## Minimum Viable Pipeline
The core API of 🤗 Diffusers is divided into three main components:

1. **Pipelines**: high-level classes designed to rapidly generate samples from popular trained diffusion models in a user-friendly fashion.
2. **Models**: popular architectures for training new diffusion models, e.g. UNet.
3. **Schedulers**: various techniques for generating images from noise during inference as well as to generate noisy images for training.

**Defining Scheduler**\
In diffusers, the scheduler handles adding noise to the input image and iteratively remove noise along with model predictions. The DDPM paper describes a corruption whose simplified form is described as follows 

$q(x_t | x_0) = N(x_t; \sqrt(ᾱ_t) x_0, (1 - ᾱ_t) I $ , where $ᾱ_t = ∏_{i=1}^T α_i$ and $α_i = 1 - β_i$

The noise addition can be configured using three parameters 

```
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.001, beta_end=0.004)
``` 
More details can be seen in `01_introduction_to_diffusers.ipynb`

**Defining Model**\
Diffusers provides us a handy `UNet2DModel` class where we need to define the `down_block_types`, `up_block_types` and `mid_block_type`.

The forward step can be done as follows:-
```
# Here model output is of UNet2DOutput class which has atrribute sample. Don't confuse it with the sampling from probability distribution. 
model_prediction = model(noisy_xb, timesteps).sample
```

**Training Loop**\


