# ThoughtsFormer
The ThoughtsFormer is a model that outputs multiple "thought" tokens before predicting its next action. 
In this repository, it is trained on next-token prediction. To predict one token, the thoughtsformer recurrently generates X thought tokens, and on the final token it generates what it believes is the correct token which acts as a reward for the model.

## Causal Masking
To ensure generated thought-tokens don't impact other trains of thought, an interesting causal method is applied, as is shown below.

![image](https://github.com/user-attachments/assets/640107bc-6678-40eb-a2e3-b0b74c6c7065)

## 2D Positional Embeddings
Using a 2D positional encoding allows our model to differentiate between position-in-sequence encodings and position-in-thought encodings. 

## Training
This model has been trained with supervised learning or reinforcement learning.
Supervised learning was successful, and the reinforcement learning method, while functional and showed results, was not as effective as supervised learning. 

### Supervised Learning
The model recurrently generates N embeddings for every single token in parallel and passes the final embedding through an output head to recieve the final token. After generating every thought embedding, the model places that embedding back into the context like this.

![image](https://github.com/user-attachments/assets/721076e7-827d-4e24-9a0d-3ac5e4970c45) 

Because the existing transformer takes input embeddings and repeatedly transforms them in the embedding space to create an embedding that encodes the necessary information to predict the next token, the ThoughtsFormer aims to allow the model to continue transforming the same embedding more in a recurrent fashion.

### Reinforcement Learning

A token-based implementation of PPO was created to train this model. The reward was defined as the cosine similarity between the final predicted token's embedding and the embedding of the label. All rewards of intermediate tokens are equal to zero. 

![image](https://github.com/user-attachments/assets/488c343d-8fcd-4ac5-a2e8-13344b33c523)

## Results

Currently, the model still performs at roughly the same level as a transformer. I'm working on some improvements that will potentially improve the architecture, but even if it isn't better than a Transformer, that's a really interesting takeaway about parameter efficiency. 

![image](https://github.com/user-attachments/assets/7754ef7b-10c7-47f1-af3a-cca123f488d0)
![image](https://github.com/user-attachments/assets/7f8add5b-4ae8-4275-94c0-e32772e5c9ba)


## Upcoming updates

This model was just updated to include supervised learning! Next up is BPTTWT (Back-Propogation Through Time Without Time)

BPTT with this context has at least quadratic scaling (because it has N thoughts per token to backpropagate and takes N timesteps to generate the sequence), but because of the causal nature of the model, the entire generation process is recreated every step of the computation because existing tokens don't percieve any difference in their computation. This means that 1) the outputs of these layers could be cached allowing a the model to never recompute hidden states (only works on small models) and 2) memory consumption is reduced from quadratic to linear because BPTT doesn't have to go through time as the entire process already exists in the final pass. This is my next area of exploration.


