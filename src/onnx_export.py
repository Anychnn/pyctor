import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelWithLMHead, BertModel

model_path = "../models/hf/corrector_900news"
tokenizer_path="../models/tokenizer/"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelWithLMHead.from_pretrained(model_path)

text = "守殊待兔"
text_inputs = tokenizer(text, return_tensors="pt")

output = model(**text_inputs)
# print(output)

text_inputs = (
    text_inputs['input_ids'],
    text_inputs['token_type_ids'],
    text_inputs['attention_mask'],
)

# export
torch.onnx.export(
    model,
    text_inputs,
    f="../models/ncnn/bert_fp32.onnx",
    input_names=['input_ids', 'token_type_ids', 'attention_mask'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {
            0: 'batch_size',
            1: 'sequence'
        },
        'token_type_ids': {
            0: 'batch_size',
            1: 'sequence'
        },
        'attention_mask': {
            0: 'batch_size',
            1: 'sequence'
        },
        'logits': {
            0: 'batch_size',
            1: 'sequence'
        }
    },
    do_constant_folding=True,
    opset_version=11,
)
