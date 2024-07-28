import torch
import numpy as np
import gc
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
    DATASET_TO_PREF,
    masked_mean
)
from torch.utils.tensorboard import SummaryWriter
import os
os.environ['CURL_CA_BUNDLE'] = ''

@dataclass
class DPOTrainingArguments(RLTrainingArguments):
    
    data_dir: str = field(default="/scratch/gpfs/cd2853/data/training_data", metadata={"help": "Data directory."})

    ipo: bool = field(default=False, metadata={"help": "Whether to use Iterative Policy Optimization loss"})

    reference_free: bool = field(default=False, metadata={"help": "Whether to use reference-model free training"})

    rpo: bool = field(default=True, metadata={"help": "Whether to use Reasoning Preference Optimization loss"})

    beta: float = field(default=0.1, metadata={"help": "Beta parameter for DPO loss"})


class DPOTrainer(RLTrainer):
    
    def rl_loss(
        self,
        step_data,
        label_smoothing: float = 0.0,
    ):  
        policy_chosen_logps = step_data["policy_chosen_logps"]
        policy_rejected_logps = step_data["policy_rejected_logps"]
        reference_chosen_logps = step_data["reference_chosen_logps"]
        reference_rejected_logps = step_data["reference_rejected_logps"]

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        
        if self.args.reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios 
        
        if self.args.ipo:
            losses = (logits - 1/(2 * self.args.beta)) ** 2 
        else:
            losses = -F.logsigmoid(self.args.beta * logits) * (1 - label_smoothing) - F.logsigmoid(-self.args.beta * logits) * label_smoothing
        
        if self.args.rpo:
            losses -= policy_chosen_logps

        return losses.mean()

    def step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        thoughts_mask: torch.Tensor,
        targets_mask: torch.Tensor,
        rewards: torch.Tensor
    ):  
        ### Select best input_ids, based on rewards
        thoughts_mask = torch.clamp(thoughts_mask + targets_mask, 0, 1)[:, 1:].contiguous()
        ### Check that rewards are well-distributed
        rewards = rewards.view(-1, 2)
        assert all(rewards[:, 0] - rewards[:, 1] >= 0), f"Rewards are not well distributed: {rewards}"
        ### Compute loss
        train_losses = []
        bs, step_size = input_ids.shape[0], self.args.mini_batch_size
        self.policy_model.train()
        for i, mini_step_start in enumerate(range(0, bs, step_size)):
            self.optimizer.zero_grad()
            mini_step_end = min(mini_step_start + step_size, bs)
            ### Select mini-batch data
            mini_input_ids = input_ids[mini_step_start:mini_step_end]
            mini_attention_mask = attention_mask[mini_step_start:mini_step_end]
            mini_thoughts_mask = thoughts_mask[mini_step_start:mini_step_end]
            ### Gather chosen and rejected logprobs
            step_data = self.select_logprobs(mini_input_ids, mini_attention_mask, mini_thoughts_mask)
            ### Compute loss
            rl_loss = self.rl_loss(step_data)
            train_losses.append(rl_loss.detach().item())
            rl_loss.backward()
            clip_grad_norm_(self.policy_model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

        ### Clear cache
        self.optimizer.zero_grad()
        gc.collect()
        torch.cuda.empty_cache()

        ### Increment scheduler
        self.scheduler.step()

        return np.mean(train_losses)

    def select_logprobs(self, input_ids, attention_mask, thoughts_mask):
        with torch.no_grad():
            ref_logprobs = self.ref_model.policy_forward(input_ids, attention_mask, thoughts_mask).logprobs
        policy_logprobs = self.policy_model.policy_forward(input_ids, attention_mask, thoughts_mask).logprobs
        ### Reshape tensors
        policy_logprobs = policy_logprobs.view(-1, 2, policy_logprobs.shape[-1]).contiguous()
        ref_logprobs = ref_logprobs.view(-1, 2, ref_logprobs.shape[-1]).contiguous()
        thoughts_mask = thoughts_mask.view(-1, 2, thoughts_mask.shape[-1]).contiguous()
        ### Select logprobs
        policy_chosen_logps = policy_logprobs[:, 0, :].contiguous()
        policy_rejected_logps = policy_logprobs[:, 1, :].contiguous()
        reference_chosen_logps = ref_logprobs[:, 0, :].contiguous()
        reference_rejected_logps = ref_logprobs[:, 1, :].contiguous()
        chosen_thoughts_mask = thoughts_mask[:, 0].contiguous()
        rejected_thoughts_mask = thoughts_mask[:, 1].contiguous()
        ### Compute masked mean
        policy_chosen_logps = masked_mean(policy_chosen_logps, chosen_thoughts_mask, axis=1)
        policy_rejected_logps = masked_mean(policy_rejected_logps, rejected_thoughts_mask, axis=1)
        reference_chosen_logps = masked_mean(reference_chosen_logps, chosen_thoughts_mask, axis=1)
        reference_rejected_logps = masked_mean(reference_rejected_logps, rejected_thoughts_mask, axis=1)
        return {
            "policy_chosen_logps": policy_chosen_logps,
            "policy_rejected_logps": policy_rejected_logps,
            "reference_chosen_logps": reference_chosen_logps,
            "reference_rejected_logps": reference_rejected_logps
        }
    
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
        for _ in range(sample_size):
            for start_i in range(0, bs, step_size):
                end_i = min(start_i + step_size, bs)
                input_str = self.policy_model.format_prompts(prompts[start_i:end_i], answers[start_i:end_i], hint=hint, datasets=datasets[start_i:end_i] if datasets else None)
                thoughts_batch, responses_batch = self.policy_model.run(input_str, format=False, log=False)
                datasets_to_add = (datasets[start_i:end_i] if datasets else ["" for _ in range(end_i - start_i)]) * (sample_size + 1)
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
        if not self.force_direct:
            new_dataframe = new_dataframe[new_dataframe['response'] != ""]
        chosen = new_dataframe.sort_values(by='reward', ascending=False).drop_duplicates(subset=['_id'], keep='first')
        rejected = new_dataframe.sort_values(by='reward', ascending=False).drop_duplicates(subset=['_id'], keep='last')
        ### Join chosen and rejected on [_id, question, answer, dataset]
        joined = pd.merge(chosen, rejected, on=['_id', 'question', 'answer', 'dataset'], suffixes=('_chosen', '_rejected'))
        new_columns = ['_id', 'question', 'answer', 'thought_chosen', 'thought_rejected', 'response_chosen', 'response_rejected', 'dataset', 'reward_chosen', 'reward_rejected']
        new_dataframe = joined[new_columns]
        new_dataframe = pd.concat([chosen, rejected]).sort_values(by='_id', ascending=True)
        new_dataset = Dataset.from_pandas(new_dataframe)
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
        try:
            new_path = self.args.data_dir + f"/train_{self.type}_epoch_{epoch}.json"
            dataset = self.load_dataset(new_path)
            df = dataset.to_pandas()
            df = df.sample(frac=1).reset_index(drop=True)
            dataset = Dataset.from_pandas(df)
            self.train_dataset = dataset
            self.train_dataloader = self.prepare_dataloader(dataset, self.data_collator, batch_size=self.args.batch_size)
            return
        except FileNotFoundError:
            pass
        self.logger.info(f"Sampling rollouts from policy model")
        self.policy_model.eval()
        new_dataset = {}
        use_hint = False
        ### Iterate over train dataloader to generate data
        self.logger.info(f"Iterating over dataloader of length: {len(self.train_dataloader)}")
        for step, batch in enumerate(tqdm(self.train_dataloader)):
            self.logger.info(f"Running step {step}/{len(self.train_dataloader)}")
            if step >= len(self.train_dataloader)//2:
                use_hint = False
                self.policy_model.few_shot_dataset = None
            batch = self.sample_rollouts(batch, hint=use_hint).to_dict()
            for k, v in batch.items():
                new_dataset[k] = new_dataset.get(k, []) + v
        new_dataset = Dataset.from_dict(new_dataset)
        ### Save dataset
        new_path = self.args.data_dir + f"/train_{self.type}_epoch_{epoch}.json"
        os.makedirs(self.args.data_dir, exist_ok=True)
        df = new_dataset.to_pandas()
        df = df.sort_values(by='_id', ascending=True)
        save_to(df, new_path)
        ### Set training dataset
        df = df.sample(frac=1).reset_index(drop=True)
        self.train_dataset = Dataset.from_pandas(df)
        self.train_dataloader = self.prepare_dataloader(new_dataset, self.data_collator, batch_size=self.args.batch_size)

    def unroll_batch(self, batch: Dict[str, Union[str, torch.Tensor]]) -> Dict[str, Union[str, torch.Tensor]]:
        ''' Expand batch to have each example in a different row'''
        df = pd.DataFrame(batch)
        df = df.sort_values(by='_id', ascending=True)
        chosen_df = df[['_id', 'question', 'answer', 'thought_chosen', 'response_chosen', 'dataset', 'reward_chosen']]
        rejected_df = df[['_id', 'question', 'answer', 'thought_rejected', 'response_rejected', 'dataset', 'reward_rejected']]
        chosen_df = chosen_df.rename(columns={'thought_chosen': 'thought', 'response_chosen': 'response', 'reward_chosen': 'reward'})
        rejected_df = rejected_df.rename(columns={'thought_rejected': 'thought', 'response_rejected': 'response', 'reward_rejected': 'reward'})
        new_df = pd.concat([chosen_df, rejected_df]).sort_values(by='_id', ascending=True)
        return new_df.to_dict()

    def train(self):
        self.logger.info(f"Starting training: {self.args.start_epoch}/{self.args.epochs}")
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.logger.info(f"Running epoch {epoch}/{self.args.epochs}")
            self.generate_data(epoch)
            total_steps = len(self.train_dataloader)
            for step, batch in enumerate(tqdm(self.train_dataloader)):
                self.logger.info(f"Running step {step}/{total_steps}")
                self.policy_model.eval()
                self.validation(step)
                self.save_checkpoint(step)
                batch = self.unroll_batch(batch)
                batch = self._format_data(batch)
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                thoughts_mask = batch['thoughts_mask']
                targets_mask = batch['target_mask']
                rewards = torch.tensor(batch['reward']).to(self.current_device)
                loss = self.step(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    thoughts_mask=thoughts_mask,
                    targets_mask=targets_mask,
                    rewards=rewards
                ) 

                ### Logging
                average_reward = rewards.mean().item()
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
                
    
        if self.args.log_with == "wandb":
            wandb.finish()
        elif self.args.log_with == "tensorboard":
            self.writer.flush()
            self.writer.close()