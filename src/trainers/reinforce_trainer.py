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
class ReinforceTrainingArguments(RLTrainingArguments):

    cliprange: float = field(default=0.2, metadata={"help": "Cliprange for logprobs."})

class ReinforceTrainer(RLTrainer):
    
    def rl_loss(self, step_data):

        rewards = step_data["rewards"]
        mask = step_data["thoughts_mask"]
        new_outputs = self.policy_model.policy_forward(
            input_ids=step_data["input_ids"], 
            attention_mask=step_data["attention_mask"],
            mask=step_data["thoughts_mask"]
        )
        
        old_logprobs = masked_mean(step_data["policy_logprobs"], mask, axis=1)
        new_logprobs = masked_mean(new_outputs.logprobs, mask, axis=1)

        ratio = torch.exp(new_logprobs - old_logprobs)
        policy_loss_1 = - rewards * ratio
        policy_loss_2 = - rewards * torch.clamp(ratio, 1 - self.args.cliprange, 1 + self.args.cliprange)
        policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

        return policy_loss

    def step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rewards: torch.Tensor,
        thoughts_mask: torch.Tensor,
        targets_mask: torch.Tensor
    ):  
        thoughts_mask = thoughts_mask[:, 1:]
        # thoughts_mask = torch.clamp(thoughts_mask + targets_mask, 0, 1)[:, 1:]
        step_data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "thoughts_mask": thoughts_mask,
            "targets_mask": targets_mask,
            }
        step_size = self.args.mini_batch_size
        bs = input_ids.shape[0]
        ### Run in batches for memory efficiency
        with torch.no_grad():
            policy_logprobs = []
            ref_logprobs = []
            for i in range(0, bs, step_size):
                j = min(i + step_size, bs)
                mini_input_ids = input_ids[i:j, :]
                mini_attention_mask = attention_mask[i:j, :]
                mini_thoughts_mask = thoughts_mask[i:j, :]
                ### Run policy model
                policy_logprobs.append(self.policy_model.policy_forward(mini_input_ids, mini_attention_mask, mini_thoughts_mask).logprobs)
                ### Run reference models
                ref_logprobs.append(self.ref_model.policy_forward(mini_input_ids, mini_attention_mask, mini_thoughts_mask).logprobs)
            policy_logprobs = torch.cat(policy_logprobs, dim=0)
            ref_logprobs = torch.cat(ref_logprobs, dim=0)
        
        ### Get advantages
        with torch.no_grad():
            rewards = self.compute_rewards(rewards, policy_logprobs, ref_logprobs, thoughts_mask)
        
        ### Update step data
        step_data["ref_logprobs"] = ref_logprobs
        step_data["policy_logprobs"] = policy_logprobs
        step_data["rewards"] = rewards
        
        ### Compute loss
        self.policy_model.train()
        train_losses = []
        bs = input_ids.shape[0]
        shuffled_indexes = np.random.permutation(bs)
        self.logger.info(f"Computing loss for {bs} samples")
        p_bar = tqdm(total=bs//step_size)
        for i, mini_step_start in enumerate(range(0, bs, step_size)):
            mini_step_end = min(mini_step_start + step_size, bs)
            mini_indexes = shuffled_indexes[mini_step_start:mini_step_end]
            self.optimizer.zero_grad()
            mini_step_data = {k: v[mini_indexes] for k, v in step_data.items()}
            rl_loss = self.rl_loss(mini_step_data)
            train_losses.append(rl_loss.detach().item())
            rl_loss.backward()
            clip_grad_norm_(self.policy_model.parameters(), 0.5)
            self.optimizer.step()
            p_bar.update()
        
        self.logger.info(f"Train losses: {train_losses}")
        ### Clear cache
        self.optimizer.zero_grad()
        del rl_loss
        del step_data
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
        new_dataset = self.compute_scores(new_dataset)
        columns_to_keep = ['_id', 'question', 'answer', 'response', 'thought', 'dataset', 'reward']
        new_dataset = {k: list(v) if isinstance(v, torch.Tensor) else v for k, v in new_dataset.items() if k in columns_to_keep}
        new_dataframe = pd.DataFrame(new_dataset)
        if not self.force_direct:
            new_dataframe = new_dataframe[new_dataframe['response'] != ""]
        new_dataset = Dataset.from_pandas(new_dataframe, preserve_index=False)
        return new_dataset

    def compute_scores(self, batch_data: Dict[str, str]):
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

    def compute_rewards(
        self,
        scores: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        ref_logprobs: torch.FloatTensor,
        masks: torch.FloatTensor,
    ):
        kl = -5e-2 * (logprobs - ref_logprobs) * masks
        non_score_reward = masked_mean(kl, masks, axis=1)
        rewards = scores + non_score_reward
        rewards = rewards.view(-1, self.sample_size)
        baseline = (rewards.sum(dim=1).unsqueeze(-1) - rewards) / (self.sample_size - 1)
        rewards = (rewards - baseline).flatten()
        return rewards
    
    def train(self):
        self.logger.info(f"Starting training: {self.args.start_epoch}/{self.args.epochs}")
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.logger.info(f"Running epoch {epoch}/{self.args.epochs}")
            total_steps = len(self.train_dataloader)
            for step, batch in enumerate(tqdm(self.train_dataloader)):
                self.logger.info(f"Running step {step}/{total_steps}")
                self.policy_model.eval()
                self.validation(step)
                self.save_checkpoint(step)
                batch = self.sample_rollouts(batch).to_dict()
                batch = self._format_data(batch)
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                thoughts_mask = batch['thoughts_mask']
                targets_mask = batch['target_mask']
                rewards = torch.tensor(batch['reward']).to(self.current_device)
                loss = self.step(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    rewards=rewards,
                    thoughts_mask=thoughts_mask,
                    targets_mask=targets_mask,
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
        self.validation(step, test=True)
                
    
        if self.args.log_with == "wandb":
            wandb.finish()
        elif self.args.log_with == "tensorboard":
            self.writer.flush()
            self.writer.close()