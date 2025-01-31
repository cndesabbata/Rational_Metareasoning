import re
import torch
import os
import pandas as pd
import logging
import torch.nn.functional as F
from typing import List
from datetime import date as d
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

######################## Splitting thoughts and answers ########################

def split_thought_and_answer(responses: List[str]) -> tuple:
    thoughts = []
    answers = []
    for response in responses:
        if "Answer:" in response:
            thought = response.split("Answer:")[0]
            thought = trim(re.sub(r'<[^>]*>', '', thought).replace("Thought:", "").strip())
            answer = response.split("Answer:")[1]
            answer = trim(re.sub(r'<[^>]*>', '', answer).split("\n")[0].strip())
        else:
            thought = "\n".join(response.split("\n")[:-1])
            answer = ""
        thoughts.append(thought)
        answers.append(answer)
    return thoughts, answers

def trim(s: str) -> str:
    ''' Remove stop words from a string.'''
    s = re.sub(r'<[^>]*>', '', s)
    s = s.replace("Thought:", "").replace("Answer:", "").strip(" \n")
    return s

######################## Compute Log probabilities from logits ########################

def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor, gather: bool = True) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=2)
    if not gather:
        return logp
    logpy = logp.gather(2, labels.unsqueeze(2)).squeeze(-1)
    return logpy

######################## Save dataframe in human-readable format ########################

def save_to(df: pd.DataFrame, path: str) -> None:
    ''' Save the dataframe to a file.'''
    folder = '/'.join(path.split('/')[:-1])
    if not os.path.exists(folder) and len(folder) > 0:
        os.makedirs(folder)
    json_file = df.to_json(orient='records', indent=4)
    with open(path, 'w') as f:
        f.write(json_file)

######################## Set logger ########################

def set_logger(logger_name:str, dir: str = None, date: str = None) -> logging.Logger:
    if not date:
        date = d.today().strftime("%Y-%m-%d")
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    log_folder = f"/mnt/u14157_ic_nlp_001_files_nfs/nlpdata1/home/desabbat/projects/Rational_Metareasoning/src/logs/{date}" if not dir else f"{dir}"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    v = 1
    log_path = f"{log_folder}/{logger_name}_{v}.log"
    print(f"Setting logger at {log_path}")
    while os.path.exists(log_path):
        v += 1
        log_path = f"{log_folder}/{logger_name}_{v}.log"
    logger.addHandler(logging.FileHandler(log_path, encoding='utf-8'))
    return logger

######################## Check answer correctness ########################

def is_correct(model_answer: str, true_answer: str) -> int:
    model_answer, true_answer = str(model_answer), str(true_answer)
    if model_answer == "":
        return 0
    model_answer = model_answer.split("Answer:")[-1].split("\n")[0].strip()
    true_answer = true_answer.split("Answer:")[-1].split("\n")[0].strip()
    return int(true_answer.lower() == model_answer.lower())

######################## Print GPU utilization ########################

def print_gpu_utilization():
    nvmlInit()
    print(f"Number of available GPUs: {torch.cuda.device_count()}.")
    print("GPU memory utilization:")
    for i in range(torch.cuda.device_count()):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU {i}: {info.used/info.total*100:.2f}% | {info.used/1024**3:.2f}GB/{info.total/1024**3:.2f}GB")

######################## Parse model output ########################

def parse_model_output(outputs: [str]) -> tuple:
    thoughts, answers = [], []
    for output in outputs:
        answer = output.split("Answer:")[-1].split("\n")[0].strip()
    return thoughts, answers
    