import torch
import numpy as np
import gc
from hydra.utils import instantiate
from trainers.rl_trainer import RLTrainer, RLTrainingArguments
from torch.nn.utils import clip_grad_norm_
import os
import logging
import torch
import warnings
import torch.nn.functional as F
import wandb
import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    DataCollator,
    get_linear_schedule_with_warmup,
    TrainingArguments
)
from reward_model import RewardModel
from policy_model import AgentPretrainedModel
from datasets import Dataset
from typing import Union, Dict, Optional
from utils import (
    is_correct,
    save_to,
    trim,
    DATASET_TO_PREF
)
from torch.utils.tensorboard import SummaryWriter
import os
os.environ['CURL_CA_BUNDLE'] = ''

@dataclass
class EITrainingArguments(RLTrainingArguments):
    
    data_dir: str = field(default="/scratch/gpfs/cd2853/data/training_data", metadata={"help": "Data directory."})


class EITrainer(RLTrainer):
    
    def sample_rollouts(self, batch_data: Dict[str, str], hint: bool = False, step: int = 0):
        self.policy_model.generation_args = {
            "pad_token_id": self.tokenizer.pad_token_id,
            "top_p": 0.9,
            "temperature": 0.5,
            "do_sample": True,
            "max_new_tokens": 512,
        }
        prompts = batch_data['question']
        answers = batch_data['answer']
        ids = batch_data['_id']
        datasets = batch_data.get('dataset', None)
        sample_size = self.args.rollout_sample_size
        new_dataset = {
        }
        bs, step_size = len(prompts), self.args.mini_batch_size
        progress_bar = tqdm(total=bs*sample_size//step_size)
        for j in range(sample_size):
            use_hint = (j % 2 == 1) and hint
            for start_i in range(0, bs, step_size):
                end_i = min(start_i + step_size, bs)
                input_str = self.policy_model.format_prompts(prompts[start_i:end_i], answers[start_i:end_i], hint=use_hint, datasets=datasets[start_i:end_i] if datasets else None)
                thoughts_batch, responses_batch = self.policy_model.run(input_str, format=False, log=False)
                datasets_to_add = (datasets[start_i:end_i] if datasets else ["" for _ in range(end_i - start_i)])
                new_dataset['_id'] = new_dataset.get('_id', []) + ids[start_i:end_i]
                new_dataset['question'] = new_dataset.get('question', []) + prompts[start_i:end_i]
                new_dataset['response'] = new_dataset.get('response', []) + responses_batch 
                new_dataset['thought'] = new_dataset.get('thought', []) + thoughts_batch
                new_dataset['dataset'] = new_dataset.get('dataset', []) + datasets_to_add
                new_dataset['answer'] = new_dataset.get('answer', []) + answers[start_i:end_i]
                progress_bar.update()
        new_dataset['_id'] = new_dataset['_id'] + ids
        new_dataset['question'] = new_dataset['question'] + prompts
        new_dataset['answer'] = new_dataset['answer'] + answers
        new_dataset['response'] = new_dataset['response'] + ["" for _ in range(len(prompts))]
        new_dataset['thought'] = new_dataset['thought'] + ["" for _ in range(len(prompts))]
        new_dataset['dataset'] = new_dataset['dataset'] + (datasets if datasets else ["" for _ in range(len(prompts))] )
        new_dataset = {k: [a[i*bs+j] for j in range(bs) for i in range(sample_size + 1)] for k, a in new_dataset.items()}
        ### Compute rewards and select best answers
        new_dataset = self.compute_rewards(new_dataset)
        columns_to_keep = ['_id', 'question', 'answer', 'response', 'thought', 'dataset', 'reward']
        new_dataset = {k: list(v) if isinstance(v, torch.Tensor) else v for k, v in new_dataset.items() if k in columns_to_keep}
        new_dataframe = pd.DataFrame(new_dataset)    
        new_dataframe = new_dataframe[new_dataframe['response'].str.len() > 0]
        ### Keep only questions with at least one correct answer
        new_dataframe['correct'] = new_dataframe.apply(lambda x: is_correct(x['response'], x['answer']), axis=1)
        correct_ids = new_dataframe[new_dataframe['correct'] > 0]['_id'].unique().tolist()
        new_dataframe = new_dataframe[new_dataframe['_id'].isin(correct_ids)]
        new_dataframe = new_dataframe[(new_dataframe['correct'] > 0) | (new_dataframe['response'] == "")]
        if not self.force_direct:
            new_dataframe = new_dataframe[new_dataframe['response'] != ""]
        new_dataframe = new_dataframe.sort_values(by='reward', ascending=False).drop_duplicates(subset=['_id'], keep='first')
        new_dataset = Dataset.from_pandas(new_dataframe, preserve_index=False)
        return new_dataset

    def compute_rewards(self, batch_data: Dict[str, str]):
        ### Compute rewards
        self.logger.info("Computing rewards")
        batch_data = self._format_data(batch_data)
        batch_rewards = []
        step_size = self.args.mini_batch_size // self.args.rollout_sample_size * (self.args.rollout_sample_size + 1)
        bs = batch_data['input_ids'].shape[0]
        ### Compute
        for i in range(0, bs, step_size):
            j = min(i + step_size, bs)
            mini_batch = {k: v[i:j] for k, v in batch_data.items()}
            mini_rewards = self.reward_model(**mini_batch)
            batch_rewards.extend(mini_rewards.tolist())
        batch_data['reward'] = batch_rewards
        return batch_data