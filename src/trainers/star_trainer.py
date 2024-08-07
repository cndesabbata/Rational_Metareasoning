import torch
import numpy as np
import gc
from trainers.rl_trainer import RLTrainer, RLTrainingArguments
from torch.nn.utils import clip_grad_norm_
from hydra.utils import instantiate
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
class StarTrainingArguments(RLTrainingArguments):
    
    data_dir: str = field(default="/scratch/gpfs/cd2853/data/training_data", metadata={"help": "Data directory."})


class StarTrainer(RLTrainer):
    
    def rl_loss(
        self,
        step_data
    ):  
        input_ids = step_data["input_ids"]
        attention_mask = step_data["attention_mask"]
        labels = step_data["labels"]
        
        output = self.policy_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = output.loss
        return loss

    def step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        thoughts_mask: torch.Tensor,
        targets_mask: torch.Tensor
    ):  
        ### Select best input_ids, based on rewards
        thoughts_mask = torch.clamp(thoughts_mask + targets_mask, 0, 1)
        labels = torch.where(thoughts_mask == 1, input_ids, torch.tensor(-100).to(self.current_device))
        ### Prepare step data
        step_data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        ### Compute loss
        self.policy_model.train()
        train_losses = []
        bs = input_ids.shape[0]
        step_size = self.args.mini_batch_size
        for _, mini_step_start in enumerate(range(0, bs, step_size)):
            mini_step_end = mini_step_start + step_size
            self.logger.info(f"Selecting mini-batch: {mini_step_start}:{mini_step_end}")
            self.optimizer.zero_grad()
            mini_batch = {k: v[mini_step_start:mini_step_end] for k, v in step_data.items()}
            loss_total = self.rl_loss(mini_batch)
            train_losses.append(loss_total.detach().item())
            loss_total.backward()
            clip_grad_norm_(self.policy_model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()

        ### Clear cache
        self.optimizer.zero_grad()
        del loss_total
        gc.collect()
        torch.cuda.empty_cache()

        ### Increment scheduler
        self.scheduler.step()

        return np.mean(train_losses)
    
    def sample_rollouts(self, batch_data: Dict[str, str], hint: bool = False):
        self.policy_model.generation_args = {
            "pad_token_id": self.tokenizer.eos_token_id,
            "top_p": 0.9,
            "temperature": 0.7,
            "do_sample": True,
            "max_new_tokens": 512,
        }
        def _sample(prompts, answers, ids, datasets=None, use_hint=False):
            new_dataset = {
            }
            bs, step_size = len(prompts), self.args.mini_batch_size*2
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
                new_dataset['reward'] = new_dataset.get('reward', []) + [is_correct(r, a) for r, a in zip(responses_batch, answers[start_i:end_i])]
            return new_dataset
        new_dataset = _sample(batch_data['question'], batch_data['answer'], batch_data['_id'], datasets=batch_data.get('dataset', None), use_hint=False)
        ### For all incorrect answers, retry with hint=True
        if hint:
            wrong_idx = [i for i, c in enumerate(new_dataset['reward']) if c==0]
            wrong_dataset = {k: [v[i] for i in wrong_idx] for k, v in new_dataset.items()}
            correct_idx = [i for i, c in enumerate(new_dataset['reward']) if c==1]
            new_dataset = {k: [v[i] for i in correct_idx] for k, v in new_dataset.items()}
            fixed_dataset = _sample(wrong_dataset['question'], wrong_dataset['answer'], wrong_dataset['_id'], datasets=wrong_dataset.get('dataset', None))
            for k, v in fixed_dataset.items():
                new_dataset[k] = new_dataset.get(k, []) + v
        ### Compute rewards and select best answers
        columns_to_keep = ['_id', 'question', 'answer', 'response', 'thought', 'dataset', 'reward']
        new_dataset = {k: list(v) if isinstance(v, torch.Tensor) else v for k, v in new_dataset.items() if k in columns_to_keep}
        new_dataframe = pd.DataFrame(new_dataset)
        new_dataframe = new_dataframe[new_dataframe['reward'] == 1]
        new_dataset = Dataset.from_pandas(new_dataframe, preserve_index=False)
        return new_dataset

    def generate_data(self, epoch: int = 0, original_df: pd.DataFrame = None):
        ### Rollouts from policy
        columns = ["_id", "question", "answer", "thought", "response", "dataset",  "reward" ]
        try:
            new_path = self.args.data_dir + f"/train_{self.policy_model.model_name}_{self.type}_epoch_{epoch}.json"
            dataset = self.load_dataset(new_path)
            df = dataset.to_pandas().sample(frac=1)[columns]
            dataset = Dataset.from_pandas(df, preserve_index=False)
            self.train_dataset = dataset
            self.train_dataloader = self.prepare_dataloader(dataset, self.data_collator, batch_size=self.args.batch_size)
            return
        except FileNotFoundError:
            pass
        self.logger.info(f"Sampling rollouts from policy model")
        self.policy_model.eval()
        new_dataset = {}
        use_hint = True
        ### Iterate over train dataloader to generate data
        if original_df is not None:
            original_dataset = Dataset.from_pandas(original_df, preserve_index=False)
            self.train_dataloader = self.prepare_dataloader(original_dataset, self.data_collator, batch_size=self.args.batch_size)
        self.logger.info(f"Iterating over dataloader of length: {len(self.train_dataloader)}")
        for step, batch in enumerate(tqdm(self.train_dataloader)):
            self.logger.info(f"Running step {step}/{len(self.train_dataloader)}")
            # if step >= len(self.train_dataloader)//2:
            #     use_hint = False
            #     self.policy_model.few_shot_dataset = None
            batch = self.sample_rollouts(batch, hint=use_hint).to_dict()
            for k, v in batch.items():
                new_dataset[k] = new_dataset.get(k, []) + v
        new_dataset = Dataset.from_dict(new_dataset)
        ### Save dataset
        new_path = self.args.data_dir + f"/train_{self.policy_model.model_name}_{self.type}_epoch_{epoch}.json"
        os.makedirs(self.args.data_dir, exist_ok=True)
        df = new_dataset.to_pandas()[columns]
        if epoch > 0:
            old_dataset = self.train_dataset.to_pandas()
            df = pd.concat([df, old_dataset]).sort_values(by='_id')
        save_to(df, new_path)
        ### Set training dataset
        df = df.sample(frac=1)
        self.train_dataset = Dataset.from_pandas(df[columns], preserve_index=False)
        self.train_dataloader = self.prepare_dataloader(new_dataset, self.data_collator, batch_size=self.args.batch_size)

    def train(self):
        self.logger.info(f"Starting training: {self.args.start_epoch}/{self.args.epochs}")
        original_df = self.train_dataset.to_pandas()
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.logger.info(f"Running epoch {epoch}/{self.args.epochs}")
            self.generate_data(epoch, original_df)
            if epoch > 0 and self.model_config is not None:
                self.reload_policy_model()
                output_suffix = f"_epoch_{epoch}" if epoch != self.args.epochs - 1 else ""
                self.output_path = "_".join(self.output_path.split("_")[:-2]) + output_suffix
            total_steps = len(self.train_dataloader)
            for step, batch in enumerate(tqdm(self.train_dataloader)):
                self.logger.info(f"Running step {step}/{total_steps}")
                self.policy_model.eval()
                self.validation(step)
                self.save_checkpoint(step)
                batch = self._format_data(batch)
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                thoughts_mask = batch['thoughts_mask']
                targets_mask = batch['target_mask']
                loss = self.step(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    thoughts_mask=thoughts_mask,
                    targets_mask=targets_mask,
                ) 

                ### Logging
                lenghts = batch['thoughts_mask'].sum(dim=1).float()
                average_length = lenghts.mean().item()
                std_length = lenghts.std().item()
                if step % self.args.logging_steps == 0:
                    if self.args.log_with == "wandb":
                        stats = {
                            'train/step': step,
                            'train/loss': abs(loss),
                            'train/average_length': average_length,
                            'train/std_length': std_length,
                        }
                        self.logger.info(f"Logging metrics: {stats}")
                        wandb.log(stats)
                    elif self.args.log_with == "tensorboard":
                        self.writer.add_scalar('train/loss', abs(loss), step)
                        self.writer.add_scalar('train/average_length', average_length, step)
                    self.training_losses[step] = loss
            self.validation(step, test=True)
    
        if self.args.log_with == "wandb":
            wandb.finish()
        elif self.args.log_with == "tensorboard":
            self.writer.flush()
            self.writer.close()