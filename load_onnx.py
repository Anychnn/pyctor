import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelWithLMHead, BertModel
import time
import onnx
import onnxruntime as rt
import numpy as  np
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelWithLMHead
import time
# model_path="bert_fp32.onnx"
model_path="/Users/bytedance/Documents/workspace/TurboTransformers/model_m.quant.onnx"
# model_path="model.onnx"
# model_path="model_m.quant.onnx"
# model_path="/opt/tiger/bytedseq/example/bert.opt.quant.onnx"

token_model_path="/Users/bytedance/Downloads/csc/tmp"
tokenizer = AutoTokenizer.from_pretrained(token_model_path)


# print(quantized_model)
# torch.save(quantized_model,"quant_bert.pt")
# text="守殊待兔"
text="你好，我今天很高性!"
text_inputs = tokenizer(text, return_tensors="pt")

inputs = {k: v.detach().cpu().numpy() for k, v in text_inputs.items()}
sess = rt.InferenceSession(model_path,providers=['CPUExecutionProvider'])
input_name0 = sess.get_inputs()[0].name
print(sess.get_inputs()[0])
# print(input_name0)
input_name1 = sess.get_inputs()[1].name
# print(input_name1)
# input_name2 = sess.get_inputs()[2].name
# print(input_name2)
output_name = sess.get_outputs()[0].name
# print(output_name)

output = sess.run([output_name], inputs)
print("输入:")
print(text)

# print(output)
output=tokenizer.convert_ids_to_tokens(np.argmax(output[0],axis=-1)[0][1:-1])
print("纠正:")
print("".join(output))
# output[0]

start=time.time()
for i in range(10):
    output = sess.run([output_name], inputs)

cost_time=(time.time()-start)/10*1000
print(f"cost {cost_time} ms")
