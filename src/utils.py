import re
import torch
import os
import pandas as pd
import logging
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from datetime import date as d

######################## Dataset prefixes ########################

COMMONSENSEQA_PREF = "Your final answer must be one of the provided options, without additional information.\n"

PROOFWRITER_PREF = "Your final answer must be a one of True/False/Unknown, without additional information.\n"

GSM8K_PREF = "Your final answer must be a number, without additional information.\n"

STRATEGYQA_PREF = "Your final answer must be either 'Yes' or 'No', without additional information.\n"

DATASET_TO_PREF = {
    "commonsenseqa": COMMONSENSEQA_PREF,
    "proofwriter": PROOFWRITER_PREF,
    "gsm8k": GSM8K_PREF,
    "strategyqa": STRATEGYQA_PREF
}

######################## Prompt Instructions ########################

DIRECT_INSTRUCTION = "Answer the following question directly. Do not include any extra information.\n"

COT_INSTRUCTION = "Answer the following question in a thoughtful way, thinking step by step to get to the solution. End your response with 'Answer: ' followed by your final answer.\n"

RMR_INSTRUCTION = "Answer the following question in a thoughtful way. You can think step by step about the problem, if needed. Otherwise, you can provide a direct answer. In either case, end your response with 'Answer: ' followed by your final answer.\n"

MODE_TO_INSTRUCTION = {
    "direct": DIRECT_INSTRUCTION,
    "cot": COT_INSTRUCTION,
    "rmr": RMR_INSTRUCTION
}

######################## Splitting thoughts and answers ########################

def split_thought_and_answer(responses: List[str]) -> tuple:
    thoughts = []
    answers = []
    for response in responses:
        if "Answer:" in response:
            thought = response.split("Answer:")[0]
            thought = re.sub(r'<[^>]*>', '', thought).replace("Thought:", "").strip()
            answer = response.split("Answer:")[1]
            answer = re.sub(r'<[^>]*>', '', answer).split("\n")[0].strip()
        elif "<thought>" in response:
            try:
                thought = trim(response.split("</thought>")[0])
                answer = trim(response.split("</thought>")[1]).split("\n")[0].strip()
            except IndexError:
                thought = trim(response)
                answer = "" 
        else:
            thought = ""
            answer = trim(response).split("\n")[0].strip()
        thoughts.append(thought)
        answers.append(answer)
    return thoughts, answers

def trim(s: str) -> str:
    ''' Remove stop words from a string.'''
    s = re.sub(r'<[^>]*>', '', s)
    s = s.replace("Thought:", "").replace("Answer:", "").strip(" \n")
    return s.strip(" \n")

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

def set_logger(logger_name:str, date: str = None) -> logging.Logger:
    if not date:
        date = d.today().strftime("%Y-%m-%d")
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    log_folder = f"logs/{date}"
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
    model_answer = re.sub(r'<.*?>', '', model_answer).split("Answer:")[-1].strip(" \n")
    true_answer = re.sub(r'<.*?>', '', true_answer).split("Answer:")[-1].strip(" \n")
    try:
        chosen_letter = re.search(r'\((.*?)\)', model_answer).group(1).strip(" \n").lower()
        target_letter = re.search(r'\((.*?)\)', true_answer).group(1).strip(" \n").lower()
        return int(chosen_letter == target_letter)
    except AttributeError:
        true_answer = re.sub(r'\(.*?\)', '', true_answer)
        model_answer = re.sub(r'\(.*?\)', '', model_answer)
        return int(str(true_answer).strip(" \n").lower() in str(model_answer).strip(" \n").lower())