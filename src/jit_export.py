import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelWithLMHead, BertModel
model_path="/opt/tiger/bytedseq/csc_filter/csc/model/selector_model_3epoch_900news"
tokenizer=AutoTokenizer.from_pretrained(model_path)
model = AutoModelWithLMHead.from_pretrained(model_path)


text="守殊待兔"
text_inputs = tokenizer(text, return_tensors="pt")

output=model(**text_inputs)
# print(output)
# a=(text_inputs['input_ids'],{
#     "token_type_ids":text_inputs['token_type_ids'],
#     "attention_mask":text_inputs['attention_mask']}
# ) 

text_inputs=(text_inputs['input_ids'],text_inputs['token_type_ids'],text_inputs['attention_mask'],)
# a=text_inputs['input_ids']


# jit
# traced_model = torch.jit.trace(quantized_model, (text_inputs['input_ids'],text_inputs['token_type_ids'],text_inputs['attention_mask']),strict=False)
# torch.jit.save(traced_model, "bert_quant.pt")


# # export
torch.onnx.export(
    model,
    text_inputs,
    f="./bert_fp32.onnx",  
    input_names=['input_ids', 'token_type_ids','attention_mask'], 
    output_names=['logits'], 
    dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'}, 
                'attention_mask': {0: 'batch_size', 1: 'sequence'}, 
                'logits': {0: 'batch_size', 1: 'sequence'}}, 
    do_constant_folding=True, 
    opset_version=11, 
)
