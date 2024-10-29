# ThoughtsFormer
The ThoughtsFormer is a model that outputs multiple "thought" tokens before predicting its next action. 
In this repository, it is trained on next-token prediction. To predict one token, the thoughtsformer recurrently generates X thought tokens, and on the final token it generates what it believes is the correct token which acts as a reward for the model.

## Causal Masking
To ensure generated thought-tokens don't impact other trains of thought, an interesting causal method is applied, as is shown below.

![image](https://github.com/user-attachments/assets/640107bc-6678-40eb-a2e3-b0b74c6c7065)

## 2D Positional Embeddings
Using a 2D positional encoding allows our model to differentiate between position-in-sequence encodings and position-in-thought encodings. 

## Training
The model recurrently generates N embeddings for every single token in parallel and passes the final embedding through an output head to recieve the final token. After generating every thought embedding, the model places that embedding back into the context like this.

![image](https://github.com/user-attachments/assets/721076e7-827d-4e24-9a0d-3ac5e4970c45) 

This work is an extension of the Universal Transformer with an added output at every step that aims to serve as a thought, or a piece of information that the model remembers and use at every subsequent thought step. 
## Results


## Upcoming updates

BPTT with this context has at least quadratic scaling (because it has N thoughts per token to backpropagate and takes N timesteps to generate the sequence), but because of the causal nature of the model, the entire generation process is recreated every step of the computation because existing tokens don't percieve any difference in their computation. This means that 1) the outputs of these layers could be cached allowing a the model to never recompute hidden states (only works on small models) and 2) memory consumption is reduced from quadratic to linear because BPTT doesn't have to go through time as the entire process already exists in the final pass. This is my next area of exploration.


