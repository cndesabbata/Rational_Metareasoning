
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
    dataset = Dataset.from_pandas(df)
    logger.info("Running model")
    def _run(row):
        input = row["question"]
        dataset = [row['dataset']] if 'dataset' in row else None
        thoughts, responses = model.run(input, datasets=dataset, log=True)
        row['thought'] = thoughts[0]
        row['response'] = responses[0]
        return row
    dataset = dataset.map(_run)
    ### Save responses
    output_dir = cfg.output_dir
    few_shot_string = "_few_shot" if model.few_shot_dataset is not None else ""
    output_path = f"{output_dir}/{model.model_name}_{model.inference_mode}{few_shot_string}.json"
    save_to(dataset.to_pandas(), output_path)

if __name__ == "__main__":
    run_model()