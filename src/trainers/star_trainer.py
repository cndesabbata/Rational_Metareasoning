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
    is_correct,
    save_to
)
import os
os.environ['CURL_CA_BUNDLE'] = ''


class StarTrainer(EITrainer):
    '''STaR Trainer'''

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
            rationalized_df = pd.DataFrame(_sample(wrong_dataloader))
            rationalized_df['with_hint'] = True
            df = df[df['correct'] == 1]
            df = pd.concat([df, rationalized_df])
            return df

        questions = batch_data['question']
        answers = batch_data['answer']
        datasets = batch_data.get('dataset', None)
        new_path = self.args.data_dir + f"/train_{self.policy_model.model_name}{self.mod_str}{self.args.output_suffix}_step_{step}_full.json"
        os.makedirs(self.args.data_dir, exist_ok=True)
        prompts = self.policy_model.format_prompts(questions, answers, hint=False, datasets=datasets)
        if hint:
            prompts_with_hint = self.policy_model.format_prompts(questions, answers, hint=True, datasets=datasets)
        dataloader = self.prepare_dataloader(Dataset.from_dict(batch_data.as_dict().copy().update({"prompts": prompts})), self.data_collator, batch_size=self.args.mini_batch_size)
        ### Sample from the policy model
        new_df = pd.DataFrame(_sample(dataloader))
        new_df['with_hint'] = False
        if hint:
            new_df = _rationalize(new_df, prompts_with_hint)
        new_df = new_df.drop(columns=['prompts'])
        ### Save the dataset
        if os.path.exists(new_path):
            os.remove(new_path)
        save_to(new_df, new_path)
        ### Filter out incorrect answers
        df = df[df['_id'].isin(df[df.apply(lambda x: is_correct(x['response'], x['answer']), axis=1) > 0]['_id'].unique())]
        return Dataset.from_pandas(df.sample(frac=1, random_state=42), preserve_index=False)

    

    