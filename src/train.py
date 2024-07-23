
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

@hydra.main(config_path="../config", config_name="train")
def train_model(cfg: DictConfig):
    print(cfg)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    logger = set_logger(logger_name="run")
    trainer = instantiate(cfg.trainer)
    trainer.set_logger(logger)
    trainer.reward_model.logger = logger
    trainer.train()
    trainer.save()

if __name__ == "__main__":
    train_model()