from transformers import AutoTokenizer, AutoConfig
import os 

tokenizer = AutoTokenizer.from_pretrained("./vocab/", tokenizer_type='bert')
sentence = [
    "We are very happy to show you the Transformers library",
    "I am not happy."
]
inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")

print(inputs)
