.no_grad():
#     tokens = torch.tensor(tokenizer(sentence)).view(1,-1)
#     custom_model.eval()
#     predictions = custom_model.forward_ppo_with_tokens(tokens, torch.zeros_like(tokens),0)[0]
  
#     # print(custom_model.transformer.transformer.layers[0](input))
    
    
#     # print(custom_model.transformer.transformer.layers[0](input))

# x = predictions.argmax(dim=-1)
# tokenizer.decod