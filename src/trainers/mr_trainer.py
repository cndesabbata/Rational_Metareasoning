import torch
from trainers.ei_trainer import EITrainer
import os
import torch
import os
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from datasets import Dataset
from typing import Dict
import numpy as np
from utils import (
    is_correct,
    save_to
)
import os
os.environ['CURL_CA_BUNDLE'] = ''


class MRTrainer(EITrainer):
    '''Metareasoning Trainer'''
    
    def sample_rollouts(self, batch_data: Dict[str, str], hint: bool = False, step: int = 0):
        
        def _sample(dataloader, rollout_step = 0):
            new_dataset = {
            }
            for mini_batch in tqdm(dataloader, desc=f"Sampling batch {rollout_step}"):
                thoughts_batch, responses_batch = self.policy_model.run(mini_batch["prompts"], log=False)
                for k, v in mini_batch.items():
                    new_dataset[k] = new_dataset.get(k, []) + v
                new_dataset['response'] = new_dataset.get('response', []) + responses_batch
                new_dataset['thought'] = new_dataset.get('thought', []) + thoughts_batch
                new_dataset['rollout_step'] = new_dataset.get('rollout_step', []) + [rollout_step] * len(responses_batch)
            return new_dataset
        
        def _rationalize(df, prompts_with_hint):
            df['correct'] = df.apply(lambda x: is_correct(x['response'], x['answer']), axis=1)
            df['prompts'] = prompts_with_hint
            wrong_dataloader = self.prepare_dataloader(Dataset.from_dict(df[df['correct'] == 0].to_dict(orient='list')), self.data_collator, batch_size=self.args.mini_batch_size)
            rationalized_df = pd.DataFrame(_sample(wrong_dataloader, rollout_step = j))
            rationalized_df['with_hint'] = True
            df = df[df['correct'] == 1]
            df = pd.concat([df, rationalized_df])
            return df

        questions = batch_data['question']
        answers = batch_data['answer']
        datasets = batch_data.get('dataset', None)
        sample_size = self.args.rollout_sample_size
        new_df = None
        new_path = self.args.data_dir + f"/train_{self.policy_model.model_name}{self.mod_str}{self.args.output_suffix}_step_{step}_full.json"
        os.makedirs(self.args.data_dir, exist_ok=True)
        prompts = self.policy_model.format_prompts(questions, answers, hint=False, datasets=datasets)
        if hint:
            prompts_with_hint = self.policy_model.format_prompts(questions, answers, hint=True, datasets=datasets)
        dataloader = self.prepare_dataloader(Dataset.from_dict(batch_data.as_dict().copy().update({"prompts": prompts})), self.data_collator, batch_size=self.args.mini_batch_size)
        if os.path.exists(new_path):
            new_df = pd.read_json(new_path)
            visited_ids = set(new_df['_id'].unique())
        else:
            visited_ids = set()
        for j in range(sample_size):
            if j in visited_ids: 
                continue
            ### Sample from the policy model
            dataset_j = _sample(dataloader, rollout_step = j)
            df_j = pd.DataFrame(dataset_j)
            df_j['with_hint'] = False
            if hint:
                df_j = _rationalize(df_j, prompts_with_hint)
            new_df = pd.concat([new_df, df_j]) if new_df is not None else df_j
            visited_ids.add(j)
            new_df = new_df.drop(columns=['prompts'])
            ### Save the dataset
            if os.path.exists(new_path):
                os.remove(new_path)
            save_to(new_df, new_path)
            
        ### Compute rewards
        direct_answer_df = pd.DataFrame(batch_data)
        direct_answer_df['with_hint'] = False
        direct_answer_df['thought'] = ""
        direct_answer_df['response'] = ""
        direct_answer_df['rollout_step'] = -1
        df = pd.concat([direct_answer_df, new_df])
        df = df.assign(thought_len=df['thought'].str.len()).sort_values(by=['_id', 'thought_len'], ascending=[True, False]).drop(columns=['thought_len'])
        df = self.compute_rewards(df)
        if not self.force_direct:
            df = df[df['response'].str.len() > 0]
        ### [Optional] Filter out incorrect answers
        df = df[df['_id'].isin(df[df.apply(lambda x: is_correct(x['response'], x['answer']), axis=1) > 0]['_id'].unique())]
        ### Compute advantages
        df['advantage'] = df['reward'] - df.groupby('_id')['reward'].transform('mean')
        df = df[df['advantage'] >= 0]
        return Dataset.from_pandas(df.sample(frac=1, random_state=42), preserve_index=False)

    def compute_rewards(self, df: pd.DataFrame) -> pd.DataFrame:
        ''' Compute rewards for the rollout dataset.'''
        def _extract_target_mask(inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
            answer_tokens = self.tokenizer.encode("Answer:", add_special_tokens=False)
            input_ids = inputs['input_ids']
            target_mask = torch.zeros_like(input_ids)
            for i, seq in enumerate(input_ids):
                start_idx = 0
                for idx in torch.where(seq == answer_tokens[0])[0]: 
                    if (seq[idx:idx+len(answer_tokens)] == answer_tokens).all():
                        start_idx = idx + len(answer_tokens)
                target_mask[i, start_idx:-1] = 1
            inputs['target_mask'] = target_mask
            return inputs
        
        def _extract_thought_mask(inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
            start_tokens = self.tokenizer.encode("<thought>", add_special_tokens=False)
            end_tokens = self.tokenizer.encode("</thought>", add_special_tokens=False)
            input_ids = inputs['input_ids']
            target_mask = torch.zeros_like(input_ids)
            for i, seq in enumerate(input_ids):
                for idx in torch.where(seq == start_tokens[0])[0]: 
                    if (seq[idx:idx+len(start_tokens)] == start_tokens).all():
                        start_idx = idx + len(start_tokens)
                for idx in torch.where(seq == end_tokens[0])[0]: 
                    if (seq[idx:idx+len(end_tokens)] == end_tokens).all():
                        end_idx = idx
                target_mask[i, start_idx:end_idx] = 1
            inputs['thoughts_mask'] = target_mask
            return inputs
        
    
        ### Compute rewards
        self.logger.info("Computing rewards")
        rewards = []
        df['text'] = df.apply(lambda x: self.policy_model.formatting_func(x)[0], axis=1)
        dataloader = self.prepare_dataloader(Dataset.from_dict(df.to_dict(orient='list')), self.completion_collator, self.args.rollout_sample_size + 1)
        for mini_batch in tqdm(dataloader, desc="Computing rewards"):
            inputs = _extract_target_mask(mini_batch)
            inputs = _extract_thought_mask(inputs)
            rewards = self.reward_model(**inputs)
            # df.loc[df['_id'].isin(mini_batch['_id']), 'reward'] = rewards
            rewards.append(pd.DataFrame(mini_batch['_id'], columns=['_id']).assign(reward=rewards))
        rewards = pd.concat(rewards, ignore_index=True)
        # merge df with rewards on _id
        df = df.merge(rewards, on='_id')
        df = df.drop(columns=['text'])
        return df
        