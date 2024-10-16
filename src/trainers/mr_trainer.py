import torch
from src.trainers.ei_trainer import EITrainer
import os
import torch
import os
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from datasets import Dataset
from typing import Dict
from utils import (
    is_correct
)
import os
os.environ['CURL_CA_BUNDLE'] = ''


class MRTrainer(EITrainer):
    '''Metareasoning Trainer'''
    
    def sample_rollouts(self, batch_data: Dict[str, str], hint: bool = False, step: int = 0):
        def _sample(prompts, answers, ids, datasets=None, use_hint=False):
            new_dataset = {
            }
            bs, step_size = len(prompts), self.args.mini_batch_size*2
            progress_bar = tqdm(total=bs//step_size)
            for start_i in tqdm(range(0, bs, step_size)):
                end_i = min(start_i + step_size, bs)
                input_str  = self.policy_model.format_prompts(prompts[start_i:end_i], answers[start_i:end_i], hint=use_hint, datasets=datasets[start_i:end_i] if datasets else None)
                thoughts_batch, responses_batch = self.policy_model.run(input_str, format=False, log=False)
                datasets_to_add = (datasets[start_i:end_i] if datasets else ["" for _ in range(end_i - start_i)]) 
                new_dataset['_id'] = new_dataset.get('_id', []) + ids[start_i:end_i]
                new_dataset['question'] = new_dataset.get('question', []) + prompts[start_i:end_i]
                new_dataset['response'] = new_dataset.get('response', []) + responses_batch 
                new_dataset['thought'] = new_dataset.get('thought', []) + thoughts_batch
                new_dataset['dataset'] = new_dataset.get('dataset', []) + datasets_to_add
                new_dataset['answer'] = new_dataset.get('answer', []) + answers[start_i:end_i]
                progress_bar.update()
                self.clear_cache()
            progress_bar.close()
            return new_dataset

        prompts = batch_data['question']
        answers = batch_data['answer']
        ids = batch_data['_id']
        datasets = batch_data.get('dataset', None)
        sample_size = self.args.rollout_sample_size
        new_dataset = {
        }
        for j in range(sample_size):
            if j == 0:
                self.policy_model.generation_args = {
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "do_sample": False,
                    "max_new_tokens": 512,
                }
            else:
                self.policy_model.generation_args = {
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "top_p": 0.9,
                    "temperature": 0.5,
                    "do_sample": True,
                    "max_new_tokens": 512,
                }
            dataset_j = _sample(prompts, answers, ids, datasets=datasets, use_hint=False)
            if hint:
                df_j = pd.DataFrame(dataset_j)
                df_j['correct'] = df_j.apply(lambda x: is_correct(x['response'], x['answer']), axis=1)
                wrong_dataset = df_j[df_j['correct'] == 0].to_dict(orient='list')
                fixed_dataset = _sample(wrong_dataset['question'], wrong_dataset['answer'], wrong_dataset['_id'], datasets=wrong_dataset.get('dataset', None), use_hint=True)
                fixed_dataset['with_hint'] = [True] * len(fixed_dataset['_id'])
                df_j = df_j[df_j['correct'] == 1]
                df_j['with_hint'] = False
                dataset_j = df_j.drop(columns=['correct']).to_dict(orient='list')
                for k, v in fixed_dataset.items():
                    dataset_j[k] = dataset_j.get(k, []) + v
            for k, v in dataset_j.items():
                new_dataset[k] = new_dataset.get(k, []) + v  
        new_dataset['_id'] = new_dataset['_id'] + ids
        new_dataset['question'] = new_dataset['question'] + prompts
        new_dataset['answer'] = new_dataset['answer'] + answers
        new_dataset['response'] = new_dataset['response'] + ["" for _ in range(len(prompts))]
        new_dataset['thought'] = new_dataset['thought'] + ["" for _ in range(len(prompts))]
        new_dataset['with_hint'] = new_dataset.get('with_hint', []) + [True for _ in range(len(prompts))]
        new_dataset['dataset'] = new_dataset['dataset'] + (datasets if datasets else ["" for _ in range(len(prompts))] )
        ### Sort by _id ascending, thought length descending
        new_dataframe = pd.DataFrame(new_dataset)
        new_dataframe['thought_len'] = new_dataframe['thought'].apply(lambda x: len(x))
        new_dataframe = new_dataframe.sort_values(by=['_id', 'thought_len'], ascending=[True, False])
        new_dataframe = new_dataframe.drop(columns=['thought_len'])
        new_dataset = new_dataframe.to_dict(orient='list')
        ### Compute rewards and select best answers
        new_dataset = self.compute_rewards(new_dataset)
        columns_to_keep = ['_id', 'question', 'answer', 'response', 'thought', 'dataset', 'reward', 'with_hint']
        new_dataset = {k: list(v) if isinstance(v, torch.Tensor) else v for k, v in new_dataset.items() if k in columns_to_keep}
        new_dataframe = pd.DataFrame(new_dataset)    
        new_dataframe['with_hint'] = new_dataframe['with_hint'].apply(lambda x: 1.0 if x else 0.0)
        if not (self.force_direct and step > 0):
            new_dataframe = new_dataframe[new_dataframe['response'].str.len() > 0]
        ### Keep only questions with at least one correct answer
        new_dataframe['correct'] = new_dataframe.apply(lambda x: is_correct(x['response'], x['answer']), axis=1)
        correct_ids = new_dataframe[new_dataframe['correct'] > 0]['_id'].unique().tolist()
        new_dataframe = new_dataframe[new_dataframe['_id'].isin(correct_ids)]
        # new_dataframe = new_dataframe[(new_dataframe['correct'] > 0) | (new_dataframe['response'] == "")]
        new_dataframe = new_dataframe[(new_dataframe['correct'] > 0) | (new_dataframe['reward'] >= 0)]
        new_dataframe = new_dataframe.sort_values(by='reward', ascending=False).drop_duplicates(subset=['_id'], keep='first')
        new_dataset = Dataset.from_pandas(new_dataframe.sample(frac=1, random_state=42), preserve_index=False)
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