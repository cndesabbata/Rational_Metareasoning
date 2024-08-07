
import pandas as pd
import logging
import os
import sys
import hydra
import torch
from reward_model import RewardModel
from hydra.utils import instantiate
from omegaconf import DictConfig
from datasets import Dataset
from utils import (
    save_to,
    set_logger
)


def compute_probabilities(
    model,
    question: str,
    answer: str,
    dataset: str
):
    formatted_prompts = model.format_prompts(
        [question],
        [answer],
        [dataset]
    )[0]
    completion = model.assistant_format.format(assistant=answer)
    print(f"Prompt: {formatted_prompts}\nCompletion: {completion}\n")
    ### Tokenize
    prompt_ids = model.tokenizer.encode(formatted_prompts, return_tensors="pt")
    completion_ids = model.tokenizer.encode(completion, return_tensors="pt")
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(model.device)
    attention_mask = torch.ones_like(input_ids).to(model.device)
    target_mask = torch.cat([torch.zeros_like(prompt_ids), torch.ones_like(completion_ids)], dim=1).to(model.device)
    print(f"Shapes: {input_ids.shape}, {attention_mask.shape}, {target_mask.shape}")
    ### Compute scores
    probs = RewardModel.compute_target_probs(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_mask=target_mask
    )
    print(f"Probs: {probs}")
    prob = probs[0]
    return prob.item()

@hydra.main(config_path="../config", config_name="run")
def run_model(cfg: DictConfig):
    print(cfg)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    logger = set_logger(logger_name="run")
    model = instantiate(cfg.model)
    model.logger = logger
    logger.info("Loading dataset")
    # print current path
    print(os.getcwd())
    df = pd.read_json(cfg.data_path)
    dataset = Dataset.from_pandas(df, preserve_index=False)
    logger.info("Running model")
    def _run(row):
        probability = compute_probabilities(
            model,
            row["question"],
            row["answer"],
            row["dataset"]
        )
        row["probability"] = probability
        return row
    dataset = dataset.map(_run)
    ### Save responses
    output_dir = cfg.output_dir
    few_shot_string = "_few_shot" if model.few_shot_dataset is not None else ""
    output_path = f"{output_dir}/probabilities_{model.model_name}_{model.inference_mode}{few_shot_string}.json"
    save_to(dataset.to_pandas(), output_path)

if __name__ == "__main__":
    run_model()