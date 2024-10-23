# ThoughtsFormer
The thoughtsformer is a model that outputs multiple "thought" tokens before predicting its next action. 
In this repository, it is trained on next-token prediction. To predict one token, the thoughtsformer recurrently generates X thought tokens, and on the final token it generates what it believes is the correct token which acts as a reward for the model.
