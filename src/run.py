
import pandas as pd
import logging
import os
import sys
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from datasets import Dataset
from utils import (
    save_to,
    set_logger
)
from tqdm import tqdm

@hydra.main(config_path="../config", config_name="run")
def run_model(cfg: DictConfig):
    print(cfg)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    logger = set_logger(logger_name="run")
    model = instantiate(cfg.model)
    model.logger = logger
    logger.info("Loading dataset")
    output_dir = cfg.output_dir
    output_suffix = ("_" + cfg.output_suffix) if cfg.output_suffix is not None else ""
    few_shot_string = "_few_shot" if model.few_shot_dataset is not None else ""
    output_path = f"{output_dir}/{model.model_name}/{model.inference_mode}{few_shot_string}{output_suffix}.json"
    os.makedirs(os.path.dirname("/".join(output_path.split("/")[:-1])), exist_ok=True)
    if os.path.exists(output_path):
        df = pd.read_json(output_path)
    else:
        df = pd.read_json(cfg.data_path)
    if all(col not in df.columns for col in ["thought", "response"]):
        df["thought"] = ""
        df["response"] = ""
    processed = df[(df["thought"] != "") & (df["response"] != "")].copy()
    logger.info(f"Already processed: {len(processed)}")
    remaining = df[(df["thought"] == "") & (df["response"] == "")].copy()
    logger.info(f"Remaining: {len(remaining)}")
    batch_size = 4
    pbar = tqdm(total=len(remaining)//batch_size)
    thoughts = []
    inputs = remaining["question"].tolist()
    datasets = remaining['dataset'].tolist() if 'dataset' in remaining else None
    for start_idx in range(0, len(remaining), batch_size):
        end_idx = min(start_idx + batch_size, len(remaining))
        batch_input = inputs[start_idx:end_idx]
        batch_dataset = datasets[start_idx:end_idx] if datasets is not None else None
        thoughts, responses = model.run(batch_input, datasets=batch_dataset, log=True)
        remaining.loc[remaining.index[start_idx:end_idx], "thought"] = thoughts
        remaining.loc[remaining.index[start_idx:end_idx], "response"] = responses
        if start_idx // batch_size % 10 == 0:
            dataset = pd.concat([processed, remaining]).sort_values("_id")
            save_to(dataset, output_path)
        pbar.update()
    ### Save responses
    dataset = pd.concat([processed, remaining]).sort_values("_id")
    save_to(dataset, output_path)

if __name__ == "__main__":
    run_model()