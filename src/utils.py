import re
import torch
import os
import pandas as pd
import logging
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional
from datetime import date as d

######################## Dataset prefixes ########################

COMMONSENSEQA_PREF = "Your final answer must be one of the provided options, without additional information. The option must be written in full, with the chosen letter in parentheses.\n"

FOLIO_PREF = "Your final answer must be a one of True/False/Unknown, without additional information.\n"

GSM8K_PREF = "Your final answer must be a number, without additional information.\n"

STRATEGYQA_PREF = "Your final answer must be either 'Yes' or 'No', without additional information.\n"

HOTPOTQA_PREF = "Your final answer must be composed of a few words, without additional information.\n"

ARC_HARD_PREF = "Your final answer must be one of the provided options, without additional information. The option must be written in full, with the chosen letter in parentheses.\n"

SOCIALIQA_PREF = "Your final answer must be one of the provided options, without additional information. The option must be written in full, with the chosen letter in parentheses.\n"

MMLU_PREF = "Your final answer must be one of the provided options, without additional information. The option must be written in full, with the chosen letter in parentheses.\n"

DATASET_TO_PREF = {
    "commonsenseqa": COMMONSENSEQA_PREF,
    "folio": FOLIO_PREF,
    "gsm8k": GSM8K_PREF,
    "strategyqa": STRATEGYQA_PREF,
    "hotpotqa": HOTPOTQA_PREF,
    "arc_hard": ARC_HARD_PREF,
    "social_iqa": SOCIALIQA_PREF,
    "mmlu": MMLU_PREF
}

######################## Prompt Instructions ########################

DIRECT_INSTRUCTION = "Answer the following question directly. Do not include any extra information.\n"

COT_INSTRUCTION = "Answer the following question in a thoughtful way, thinking step by step to get to the solution. End your response with 'Answer: ' followed by your final answer.\n"

RMR_INSTRUCTION = "Answer the following question in a thoughtful way. You can think step by step about the problem, if needed (if not, you can answer directly). End your response with 'Answer: ' followed by your final answer.\n"

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
            thought = trim(re.sub(r'<[^>]*>', '', thought).replace("Thought:", "").strip())
            answer = response.split("Answer:")[1]
            answer = trim(re.sub(r'<[^>]*>', '', answer).split("\n")[0].strip())
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
    log_folder = f"logs/{date}" if not dir else f"{dir}"
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
    if model_answer == "":
        return 0
    model_answer = model_answer.split("Answer:")[-1].split("\n")[0].strip()
    true_answer = true_answer.split("Answer:")[-1].split("\n")[0].strip()
    return int(true_answer.lower() == model_answer.lower())
    # try:
    #     chosen_letter = re.search(r'\((.*?)\)', model_answer).group(1).strip(" \n").lower()
    #     target_letter = re.search(r'\((.*?)\)', true_answer).group(1).strip(" \n").lower()
    #     return int(chosen_letter == target_letter)
    # except AttributeError:
    #     true_answer = re.sub(r'\(.*?\)', '', true_answer)
    #     model_answer = re.sub(r'\(.*?\)', '', model_answer)
    #     return int(str(true_answer).strip(" \n").lower() in str(model_answer).strip(" \n").lower())
    

######################## Masked Operations ########################
    
def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: Optional[bool] = None) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    mask = mask.float()
    # Avoid division by zero
    if axis is not None:
        return (values * mask).sum(axis=axis) / torch.clamp(mask.sum(axis=axis), min=1)
    else:
        return (values * mask).sum() / torch.clamp(mask.sum(), min=1)

def masked_var(values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance

def clip_by_value(x: torch.Tensor, tensor_min: float, tensor_max: float) -> torch.Tensor:
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped

def masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened

######################## Plot Scores ########################

def print_scores(path, tokenizer, subset: str = None):
    df = pd.read_json(path)
    if subset is not None:
        df = df[df['dataset'].str.contains(subset)]
    df['response'] = df['response'].apply(lambda x: x.split("Answer:")[-1].strip())
    df['score'] = df.apply(lambda x: is_correct(x['response'], x['answer']), axis=1)
    print(f"Accuracy: {df['score'].mean():.2f}")
    if subset is None:
        print("Per dataset accuracy:" + f"\n{df.groupby('dataset')['score'].mean()}")
    if 'few_shot' in path:
        few_shot_path = "/home/cd2853/rational_metareasoning/data/few_shot_prompts.json"
        few_shot_dataset = pd.read_json(few_shot_path)
        if 'direct' in path:
            few_shot_dataset['full_example'] = few_shot_dataset.apply(lambda x: f"{x['user']}\nAnswer: {x['answer']}", axis=1)
        else:
            few_shot_dataset['full_example'] = few_shot_dataset.apply(lambda x: f"{x['user']}\n{x['thought']}\nAnswer: {x['answer']}", axis=1)
        few_shot_dataset['tokenized_example'] = few_shot_dataset['full_example'].apply(lambda x: tokenizer(x, return_tensors="pt", padding=True, truncation=True)["input_ids"])
        few_shot_dataset['input_length'] = few_shot_dataset['tokenized_example'].apply(lambda x: x.shape[1])
        few_shot_dataset = few_shot_dataset[['dataset', 'input_length']].groupby('dataset').sum()
        df = pd.merge(df, few_shot_dataset, on='dataset')
    else:
        df['tokenized_input'] = df['question'].apply(lambda x: tokenizer(x, return_tensors="pt", padding=True, truncation=True)["input_ids"])
        df['input_length'] = df['tokenized_input'].apply(lambda x: x.shape[1])
    print(f"\nAverage length of inputs: {df['input_length'].mean():.2f}")
    if subset is None:
        print("Per dataset average length of inputs:" + f"\n{df.groupby('dataset')['input_length'].mean()}")
    df['tokenized_thought'] = df['thought'].apply(lambda x: tokenizer(x, return_tensors="pt", padding=True, truncation=True)["input_ids"])
    df['output_length'] = df['tokenized_thought'].apply(lambda x: x.shape[1])
    print(f"\nAverage length of thoughts: {df['output_length'].mean():.2f}")
    if subset is None:
        print("Per dataset average length of thoughts:" + f"\n{df.groupby('dataset')['output_length'].mean()}")

######################## Latex template table ########################
        

# \begin{tabular}\{|c|l|c|c|c|c|c|\}
# \hline
# \multirow{2}{*}{\textbf{Model}} & \multirow{2}{*}{\textbf{Setting}} & \multicolumn{3}{c|}{\textbf{Performance Metrics}} \\ \cline{3-5} 
#  &  & Avg. Input Length & Avg. Output Length & Avg. Accuracy \\ \hline
# \multirow{5}{*}
# {\textbf{Phi2}}
#  & Direct Few Shot & 278.0 & 0.0 & 63.0\% \\ \cline{2-5} 
#  & CoT Few Shot & 735.0 & 75.2 & 57.2\% \\ \cline{2-5} 
#  & RMR Few Shot & 735.0 & 79.5 & 58.8\% \\ \cline{2-5}
#  & STaR & 44.71 & 70.3 & 60.0\% \\ \cline{2-5} 
#  & RMR Training & 44.71 & - & -\% \\ \hline
# \end{tabular}
# """

table_template_non_instruct = """
\\begin{{tabular}}{{|c|l|c|c|c|c|c|}}
\\hline
\\multirow{{2}}{{*}}{{\\textbf{{Model}}}} & \\multirow{{2}}{{*}}{{\\textbf{{Setting}}}} & \\multicolumn{{3}}{{c|}}{{\\textbf{{Performance Metrics}}}} \\\\ \\cline{{3-5}} 
 &  & Avg. Input Length & Avg. Output Length & Avg. Accuracy \\\\ \\hline
\\multirow{{4}}{{*}}
{{\\textbf{{Meta-Llama-3-8B}}}}
    & Direct Few Shot & {direct_few_shot_input} & {direct_few_shot_output} & {direct_few_shot_accuracy}\% \\\\ \\cline{{2-5}}
    & CoT Few Shot & {cot_few_shot_input} & {cot_few_shot_output} & {cot_few_shot_accuracy}\% \\\\ \\cline{{2-5}}
    & STaR & {star_input} & {star_output} & {star_accuracy}\% \\\\ \\cline{{2-5}}
    & RMR Training & {rmr_training_input} & {rmr_training_output} & {rmr_training_accuracy}\% \\\\ \\hline
\\end{{tabular}}
"""

table_template_instruct = """
\\begin{{tabular}}{{|c|l|c|c|c|c|c|}}
\\hline
\\multirow{{2}}{{*}}{{\\textbf{{Model}}}} & \\multirow{{2}}{{*}}{{\\textbf{{Setting}}}} & \\multicolumn{{3}}{{c|}}{{\\textbf{{Performance Metrics}}}} \\\\ \\cline{{3-5}} 
 &  & Avg. Input Length & Avg. Output Length & Avg. Accuracy \\\\ \\hline
\\multirow{{8}}{{*}}
{{\\textbf{{Llama-3-8B-I}}}}
    & Direct & {direct_input} & {direct_output} & {direct_accuracy}\% \\\\ \\cline{{2-5}}
    & Direct Few Shot & {direct_few_shot_input} & {direct_few_shot_output} & {direct_few_shot_accuracy}\% \\\\ \\cline{{2-5}}
    & CoT & {cot_input} & {cot_output} & {cot_accuracy}\% \\\\ \\cline{{2-5}}
    & CoT Few Shot & {cot_few_shot_input} & {cot_few_shot_output} & {cot_few_shot_accuracy}\% \\\\ \\cline{{2-5}}
    & RMR & {rmr_input} & {rmr_output} & {rmr_accuracy}\% \\\\ \\cline{{2-5}}
    & RMR Few Shot & {rmr_few_shot_input} & {rmr_few_shot_output} & {rmr_few_shot_accuracy}\% \\\\ \\cline{{2-5}}
    & STaR & {star_input} & {star_output} & {star_accuracy}\% \\\\ \\cline{{2-5}}
    & RMR Training & {rmr_training_input} & {rmr_training_output} & {rmr_training_accuracy}\% \\\\ \\hline
\\end{{tabular}}
"""

def fill_table(dir, subset, tokenizer):
    # if model is not instruct, 
    if True:
        suffixes = {
            "direct_few_shot": "direct_few_shot",
            "cot_few_shot": "cot_few_shot",
            "star": "star",
            "rmr_training": "ei"
        }
        table_template = table_template_non_instruct
    results = {}
    for key, value in suffixes.items():
        try:
            path = f"{dir}/{value}.json"
            df = pd.read_json(path)
            df = df[df['dataset'].str.contains(subset)]
            df['response'] = df['response'].apply(lambda x: x.split("Answer:")[-1].strip())
            df['score'] = df.apply(lambda x: is_correct(x['response'], x['answer']), axis=1)
            few_shot_path = "/home/cd2853/rational_metareasoning/data/few_shot_prompts.json"
            df['tokenized_thought'] = df['thought'].apply(lambda x: tokenizer(x, return_tensors="pt", padding=True, truncation=True)["input_ids"])
            df['output_length'] = df['tokenized_thought'].apply(lambda x: x.shape[1])
            df['tokenized_question'] = df['question'].apply(lambda x: tokenizer(x, return_tensors="pt", padding=True, truncation=True)["input_ids"])
            df['question_length'] = df['tokenized_question'].apply(lambda x: x.shape[1])
            df['input_length'] = df['question_length']
            if 'few_shot' in value:
                few_shot_dataset = pd.read_json(few_shot_path)
                if 'direct' in path:
                    few_shot_dataset['full_example'] = few_shot_dataset.apply(lambda x: f"{x['user']}\nAnswer: {x['answer']}", axis=1)
                else:
                    few_shot_dataset['full_example'] = few_shot_dataset.apply(lambda x: f"{x['user']}\n{x['thought']}\nAnswer: {x['answer']}", axis=1)
                few_shot_dataset['tokenized_example'] = few_shot_dataset['full_example'].apply(lambda x: tokenizer(x, return_tensors="pt", padding=True, truncation=True)["input_ids"])
                few_shot_dataset['examples_length'] = few_shot_dataset['tokenized_example'].apply(lambda x: x.shape[1])
                few_shot_dataset = few_shot_dataset[['dataset', 'examples_length']].groupby('dataset').sum()
                df = pd.merge(df, few_shot_dataset, on='dataset')
                df['input_length'] = df['input_length'] + df['examples_length']
            results[f"{key}_input"] = (df['input_length'].mean(), ".1f")
            results[f"{key}_output"] = (df['output_length'].mean(), ".1f")
            results[f"{key}_accuracy"] = (df['score'].mean() * 100, ".1f")
        except FileNotFoundError:
            results[f"{key}_input"] = (0, ".1f")
            results[f"{key}_output"] = (0, ".1f")
            results[f"{key}_accuracy"] = (0, ".1f")
    return table_template.format(**{k: f"{v[0]:{v[1]}}" for k, v in results.items()})
            