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
from utils import (
    is_correct
)
import os
os.environ['CURL_CA_BUNDLE'] = ''


class StarTrainer(EITrainer):
    '''STaR Trainer'''

    def sample_rollouts(self, batch_data: Dict[str, str], hint: bool = False, step: int = 0):
        self.policy_model.generation_args = {
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": False,
            "max_new_tokens": 512,
        }
        def _sample(prompts, answers, ids, datasets=None, use_hint=False):
            new_dataset = {
            }
            bs, step_size = len(prompts), self.args.mini_batch_size*2
            for start_i in tqdm(range(0, bs, step_size)):
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
                self.clear_cache()
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

    

    