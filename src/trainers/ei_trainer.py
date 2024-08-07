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
        prompts = batch_data['question']
        answers = batch_data['answer']
        ids = batch_data['_id']
        datasets = batch_data.get('dataset', None)
        sample_size = self.args.rollout_sample_size
        new_dataset = {
        }
        bs, step_size = len(prompts), self.args.mini_batch_size*2
        progress_bar = tqdm(total=bs*sample_size//step_size)
        for j in range(sample_size):
            use_hint = j == 0 and hint
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
        ### Keep only questions with at least one correct answer
        new_dataframe['correct'] = new_dataframe.apply(lambda x: is_correct(x['response'], x['answer']), axis=1)
        correct_df = new_dataframe[new_dataframe['correct'] > 0]['_id'].unique()
        new_dataframe = new_dataframe[new_dataframe['_id'].isin(correct_df)]
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
    
    def generate_data(self, epoch: int = 0):
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
        df = self.train_dataset.to_pandas()
        df = df.sort_values(by='_id').drop_duplicates(subset=['_id'], keep='first')[['_id', 'question', 'answer', 'dataset']]
        train_dataset = Dataset.from_pandas(df, preserve_index=False)
        train_dataloader = self.prepare_dataloader(train_dataset, self.data_collator, batch_size=self.args.batch_size)
        self.logger.info(f"Iterating over dataloader of length: {len(train_dataloader)}")
        for step, batch in enumerate(tqdm(train_dataloader)):
            self.logger.info(f"Running step {step}/{len(train_dataloader)}")
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
            df = pd.concat([df, old_dataset]).sort_values(by='_id')[columns]
        save_to(df, new_path)
        ### Set training dataset
        df = df.sample(frac=1)
        self.train_dataset = Dataset.from_pandas(df, preserve_index=False)
        self.train_dataloader = self.prepare_dataloader(new_dataset, self.data_collator, batch_size=self.args.batch_size)

    def train(self):
        self.logger.info(f"Starting training: {self.args.start_epoch}/{self.args.epochs}")
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.logger.info(f"Running epoch {epoch}/{self.args.epochs}")
            self.generate_data(epoch)
            if epoch > 0 and self.model_config is not None:
                self.logger.info("Reloading policy model")
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
                average_reward = torch.tensor(batch['reward']).mean().item()
                lenghts = batch['thoughts_mask'].sum(dim=1).float()
                average_length = lenghts.mean().item()
                std_length = lenghts.std().item()
                if step % self.args.logging_steps == 0:
                    if self.args.log_with == "wandb":
                        stats = {
                            'train/step': step,
                            'train/loss': abs(loss),
                            'train/average_length': average_length,
                            'train/average_reward': average_reward,
                            'train/std_length': std_length,
                        }
                        self.logger.info(f"Logging metrics: {stats}")
                        wandb.log(stats)
                    elif self.args.log_with == "tensorboard":
                        self.writer.add_scalar('train/loss', abs(loss), step)
                        self.writer.add_scalar('train/average_length', average_length, step)
                        self.writer.add_scalar('train/average_reward', average_reward, step)
                    self.training_losses[step] = loss
            self.validation(step, test=True)
    
        if self.args.log_with == "wandb":
            wandb.finish()
        elif self.args.log_with == "tensorboard":
            self.writer.flush()
            self.writer.close()