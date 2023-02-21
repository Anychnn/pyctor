import hashlib
import requests
from tqdm import tqdm
import shutil

def download(url:str, save_path:str, sha256:str, length:str) -> bool:
    '''
    Download a file from url to save_path.
    If the file already exists, check its md5.
    If the md5 matches, return True,if the md5 doesn't match, return False.
    :param url: the url of the file to download
    :param save_path: the path to save the file
    :param md5: the md5 of the file
    :param length: the length of the file
    :return: True if the file is downloaded successfully, False otherwise
    '''

    try:
        response = requests.get(url=url, stream=True)
        with open(save_path, "wb") as f:
            with tqdm.wrapattr(response.raw, "read", total=length, desc="Downloading") as r_raw:
                shutil.copyfileobj(r_raw, f) 
    
        return True if hashlib.sha256(open(save_path, "rb").read()).hexdigest() == sha256 else False
    except Exception as e:
        print(e)
        return False


# import os
# model_size=126979102
# # print(os.path.getsize("/Users/bytedance/Documents/workspace/pyctor/models/ncnn/corrector.quant.onnx"))
# model_url="https://huggingface.co/anyang/bert_chinese_corrector_ncnn/resolve/main/corrector.quant.onnx"
# model_name="corrector.quant.onnx"
# model_sha="d9fc70641f6c938de203989dc819edb01e02259305b4c25b270ccafe41adde00"
# save_path=f"../models/ncnn/{model_name}"

# result=download(model_url,save_path,model_sha,model_size)
# # download_file(model_url,save_path)
# print(result)
# # def download():
    
