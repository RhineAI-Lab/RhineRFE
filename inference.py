import time
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

print('Start at:', time.time())

model_path = "./model/Qwen/Qwen2___5-0___5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map="cuda:0",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

print('Loaded model at:', time.time())

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "写一篇transformer结构的技术报告1000字"}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print('Finished at:', time.time())
print()
print(response)
print()
print('Text length:', len(text))
