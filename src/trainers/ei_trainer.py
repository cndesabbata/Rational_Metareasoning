import torch
import torch.nn.functional as F
import os
import pandas as pd
import gc
import numpy as np
from hydra.utils import instantiate
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
    TrainingArguments
)
from reward_model import RewardModel
from policy_model import PretrainedModel
from datasets import Dataset
from typing import Union, Dict, Optional, List, Any
from utils import (
    is_correct,
    save_to,
    trim
)
import os
os.environ['CURL_CA_BUNDLE'] = ''

@dataclass
class EITrainingArguments(TrainingArguments):

    batch_size: int = field(default=4, metadata={"help": "Batch size for data loader."})
    
    mini_batch_size: int = field(default=4, metadata={"help": "Mini batch size for rl training."})
    
    learning_rate: float = field(default=1e-4, metadata={"help": "Learning rate for optimizer."})

    start_step: int = field(default=0, metadata={"help": "Step from which to start training."})
    
    output_dir: str = field(default=None, metadata={"help": "Output directory."})

    output_suffix: str = field(default="", metadata={"help": "Output suffix."})

    rollout_sample_size: int = field(default=4, metadata={"help": "Number of rollouts."})

    voc_gamma: float = field(default=5e-4, metadata={"help": "Value of computation gamma."})  

    logging_steps: int = field(default=10, metadata={"help": "Logging steps."}) 

    save_steps: int = field(default=10, metadata={"help": "Save steps."})

    eval_steps: int = field(default=10, metadata={"help": "Evaluation steps."})

    warmup_ratio: float = field(default=0.01, metadata={"help": "Warmup ratio."})

    use_hint: bool = field(default=True, metadata={"help": "Use hint for training."})

    data_dir: str = field(default="", metadata={"help": "Data directory."})


class CustomEIDataCollator(DataCollatorForLanguageModeling):
        
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        if "input_ids" in examples[0]:
            batch_longest = {k: max([len(e[k]) for e in examples]) for k in examples[0].keys()}
            batch = {
                k: torch.stack([F.pad(torch.tensor(e[k]), (0, batch_longest[k] - len(e[k])), value = (self.tokenizer.pad_token_id if "ids" in k else 0)) for e in examples]) for k in examples[0].keys()
            }
        else:
            keys = examples[0].keys()
            batch = {
                k: [e[k] for e in examples] for k in keys
            }
        return batch

class EITrainer():
    '''Expert Iteration Trainer'''

    def __init__(self,
            model: PretrainedModel,
            args: EITrainingArguments,
            train_dataset_path: str,
            eval_dataset_path: str = None,
            test_dataset_path: str = None,
            max_seq_length: Optional[int] = None,
            type: str = "",
            force_direct: bool = False
        ):
        self.logger = None
        ### Training arguments
        self.args = args
        self.current_device = model.device
        self.tokenizer = model.tokenizer
        self.max_seq_length = max_seq_length or self.tokenizer.model_max_length
        self.training_losses = {}
        self.validation_losses = {}
        self.validation_accs = {}
        self.checkpoint_losses = {}
        self.checkpoint_accs = {}
        self.type = type
        self.force_direct = force_direct
        self.sample_size = args.rollout_sample_size
        if force_direct:
            self.sample_size += 1
        self.args.mini_batch_size = (self.args.mini_batch_size // self.sample_size) * self.sample_size
        ### Models
        self.policy_model = model
        self.model_config = None
        self.reward_model = RewardModel(
            model=self.policy_model,
            voc_gamma=self.args.voc_gamma,
            logger=self.logger,
            sample_size=self.args.rollout_sample_size,
            device=self.current_device,
            tokenizer=self.tokenizer)
        self.thought_format = self.policy_model.thought_format
        self.user_format = self.policy_model.user_format
        self.assistant_format = self.policy_model.assistant_format
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        ### Dataloaders
        self.train_dataset = self.load_dataset(train_dataset_path)
        self.eval_dataset = self.load_dataset(eval_dataset_path) if eval_dataset_path else None
        self.test_dataset = self.load_dataset(test_dataset_path) if test_dataset_path else None
        data_collator = CustomEIDataCollator(tokenizer=self.tokenizer, mlm=False)
        self.train_dataloader = self.prepare_dataloader(self.train_dataset, data_collator, batch_size=self.args.batch_size)
        self.eval_dataloader = self.prepare_dataloader(self.eval_dataset, data_collator, batch_size=self.args.mini_batch_size) if self.eval_dataset else None
        self.test_dataloader = self.prepare_dataloader(self.test_dataset, data_collator, batch_size=self.args.mini_batch_size) if self.test_dataset else None
        self.data_collator = data_collator
        ### Optimizer 
        self.set_optimizer()
        self.mod_str = "_direct" if self.force_direct else ""
        self.output_path = self.args.output_dir + self.policy_model.model_name.split("/")[-1] + self.mod_str + self.args.output_suffix + "_step_0"

    def set_logger(self, logger):
        '''Set logger'''
        self.logger = logger
        self.reward_model.logger = logger
        self.policy_model.logger = logger

    def set_optimizer(self):
        '''Initialize optimizer and scheduler'''
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.policy_model.parameters()), lr=self.args.learning_rate)
        self.total_steps = len(self.train_dataset) // self.args.mini_batch_size
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=int(self.args.warmup_ratio * self.total_steps), num_training_steps=self.total_steps)
        
    def load_policy_model(self, step=0):
        '''Load policy model at a specific step. Can be used to load a checkpoint.'''
        self.logger.info(f"Loading model at step {step}")
        if step != 0:
            path = "_".join(self.output_path.split("_")[:-2]) + f"_step_{step-1}"
            self.model_config.config.model_name_or_path = path
        self.policy_model = instantiate(self.model_config)
        self.policy_model.logger = self.logger
        self.reward_model = RewardModel(
            model=self.policy_model,
            voc_gamma=self.args.voc_gamma,
            logger=self.logger,
            sample_size=self.args.rollout_sample_size,
            device=self.current_device,
            tokenizer=self.tokenizer)
        self.set_optimizer()
        self.policy_model.model_name = self.policy_model.model_name.split("/")[-1].split("_")[0]
        self.clear_cache()

    def load_dataset(self, path: str, shuffle=True):
        '''Load dataset from path
        Args:
            path (str): Path to dataset
        Returns:
            dataset (Dataset): Dataset object
        '''
        try:
            df = pd.read_json(path)
            if shuffle:
                df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            dataset = Dataset.from_pandas(df, preserve_index=False)
        except:
            # print current path
            print(os.getcwd())
            raise FileNotFoundError(f"Dataset not found at {path}")
        return dataset

    def prepare_dataloader(self, dataset: Union[torch.utils.data.Dataset, Dataset], data_collator=None, batch_size=None):
        '''Prepare dataloader from dataset
        Args:
            dataset (Dataset): Dataset object
            data_collator (DataCollator): Data collator
            batch_size (int): Batch size
        Returns:
            dataloader (DataLoader): DataLoader object
        '''
        batch_size = batch_size or self.args.batch_size
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=data_collator,
            shuffle=False,
            drop_last=True
        )
        return dataloader

    def clear_cache(self):
        '''Clear cache'''
        self.optimizer.zero_grad()
        gc.collect()
        torch.cuda.empty_cache()

    def save(self):
        '''Save model'''
        self.policy_model.save_pretrained(self.output_path)
        self.tokenizer.save_pretrained(self.output_path)
        self.logger.info(f"Model saved at {self.output_path}")
    
    def cross_entropy_loss(self, input_ids, attention_mask, targets_mask):
        '''Compute cross entropy loss
        Args:
            input_ids (torch.Tensor): tokenized input 
            attention_mask (torch.Tensor): attention mask
            targets_mask (torch.Tensor): mask to identify labels
        Returns:
            loss (torch.Tensor): Cross entropy loss
        '''
        labels = torch.where(targets_mask.bool(), input_ids, -100*torch.ones_like(input_ids))
        outputs = self.policy_model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss

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
        ### Clear cache
        self.clear_cache()
        for _, mini_step_start in enumerate(range(0, bs, step_size)):
            mini_step_end = mini_step_start + step_size
            self.logger.info(f"Selecting mini-batch: {mini_step_start}:{mini_step_end}")
            mini_batch = {k: v[mini_step_start:mini_step_end] for k, v in step_data.items()}
            loss_total = self.rl_loss(mini_batch)
            train_losses.append(loss_total.detach().item())
            loss_total.backward()
            self.optimizer.step()
            self.scheduler.step()
            ### Clear cache
            del loss_total
            self.clear_cache()

        ### Increment scheduler
        self.scheduler.step()

        return np.mean(train_losses)

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

    def _format_data(self, data):
        ### Prepare prompts, answers, thoughts and responses
        prompts = [trim(p) for p in data['question']]
        answers = [trim(c) for c in data['answer']]
        responses = [trim(a) for a in data['response']]
        thoughts = [trim(t) for t in data['thought']]
        prompts = [self.bos_token + self.user_format.format(user=p) for p in prompts]
        answers = [self.assistant_format.format(assistant=c) + self.eos_token for c in answers]
        responses = [self.assistant_format.format(assistant=a) + self.eos_token for a in responses]
        thoughts = [self.thought_format.format(thought=t) if t else "" for t in thoughts]
        ### Tokenize
        prompt_ids = [self.tokenizer.encode(p, return_tensors="pt", add_special_tokens=False).squeeze(0) for p in prompts]
        answer_ids = [self.tokenizer.encode(c, return_tensors="pt", add_special_tokens=False).squeeze(0) for c in answers]
        thoughts_ids = [self.tokenizer.encode(t, return_tensors="pt", add_special_tokens=False).squeeze(0) for t in thoughts]
        responses_ids = [self.tokenizer.encode(a, return_tensors="pt", add_special_tokens=False).squeeze(0) for a in responses]
        ### Prepare input ids, attention mask and response mask
        input_ids = [torch.cat([p, t, c], dim=0) for p, t, c in zip(prompt_ids, thoughts_ids, answer_ids)]
        attention_mask = [torch.cat([torch.ones_like(p), torch.ones_like(t), torch.ones_like(c)], dim=0) for p, t, c in zip(prompt_ids, thoughts_ids, answer_ids)]
        thoughts_mask = [torch.cat([torch.zeros_like(p), torch.ones_like(t), torch.zeros_like(c)], dim=0) for p, t, c in zip(prompt_ids, thoughts_ids, answer_ids)]
        target_mask = [torch.cat([torch.zeros_like(p), torch.zeros_like(t), torch.ones_like(c)], dim=0) for p, t, c in zip(prompt_ids, thoughts_ids, answer_ids)]
        responses_mask = [torch.ones_like(a) for a in responses_ids]
        ### Padding
        longest = max([len(i) for i in input_ids])
        input_ids = torch.stack([F.pad(i, (0, longest - len(i)), value=self.tokenizer.pad_token_id) for i in input_ids])
        attention_mask = torch.stack([F.pad(i, (0, longest - len(i)), value=0) for i in attention_mask])
        thoughts_mask = torch.stack([F.pad(i, (0, longest - len(i)), value=0) for i in thoughts_mask])
        target_mask = torch.stack([F.pad(i, (0, longest - len(i)), value=0) for i in target_mask])
        longest_responses = max([len(i) for i in responses_ids])
        responses_ids = torch.stack([F.pad(i, (0, longest_responses - len(i)), value=self.tokenizer.pad_token_id) for i in responses_ids])
        responses_mask = torch.stack([F.pad(i, (0, longest_responses - len(i)), value=0) for i in responses_mask])
        new_data = {
            "input_ids": input_ids.long().to(self.policy_model.device),
            "attention_mask": attention_mask.long().to(self.policy_model.device),
            "thoughts_mask": thoughts_mask.long().to(self.policy_model.device),
            "target_mask": target_mask.long().to(self.policy_model.device)
            }
        data = {**data, **new_data}
        return data

    def sample_rollouts(self, batch_data: Dict[str, str], hint: bool = False):
        raise NotImplementedError

    def generate_data(self, step: int = 0, batch: Dict[str, str] = None):
        ### Rollouts from policy
        columns = ["_id", "question", "answer", "thought", "response", "dataset", "reward"]
        try:
            new_path = self.args.data_dir + f"/train_{self.policy_model.model_name}{self.mod_str}{self.args.output_suffix}_step_{step}.json"
            dataset = self.load_dataset(new_path, shuffle=False)
            return dataset.to_dict()
        except FileNotFoundError:
            pass
        self.logger.info(f"Sampling rollouts from policy model")
        use_hint = self.args.use_hint
        ### Iterate over batch to generate data
        new_batch = self.sample_rollouts(batch, hint=use_hint, step=step).to_dict()
        new_dataset = Dataset.from_dict(new_batch)
        ### Save dataset
        new_path = self.args.data_dir + f"/train_{self.policy_model.model_name}{self.mod_str}{self.args.output_suffix}_step_{step}.json"
        os.makedirs(self.args.data_dir, exist_ok=True)
        df = new_dataset.to_pandas()[columns].sample(frac=1, random_state=42).reset_index(drop=True)
        save_to(df, new_path)
        return new_dataset.to_dict()

    def train(self):
        self.logger.info(f"Starting training: {self.args.start_step}/{len(self.train_dataloader)}")
        previous_batch = None
        for step, batch in enumerate(tqdm(self.train_dataloader)):
            self.logger.info(f"Running step {step}/{len(self.train_dataloader)}")
            previous_batch = {k: v + previous_batch[k] for k, v in batch.items()} if previous_batch is not None else batch
            if self.args.start_step > step:
                continue
            self.policy_model.eval()
            if step == self.args.start_step:
                self.load_policy_model(step)
            new_data = self.generate_data(step, previous_batch)
            batch = new_data
            ### Reset model
            self.load_policy_model(step=0)
            self.output_path = "_".join(self.output_path.split("_")[:-2]) + f"_step_{step}"
            self.policy_model.train()
            ### Train model
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
            ### Validation and Saving
            self.validation(step)
            self.save_checkpoint(step)

    def validation(self, step, test=False):
        '''Validation step'''
        if self.eval_dataloader is None and not test:
            self.logger.info("No evaluation dataset provided")
            return
        self.logger.info(f"Step {step}: Validation | Test {test}")
        if step % self.args.eval_steps != 0 and not test:
            return
        self.policy_model.generation_args = {
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": False,
            "max_new_tokens": 512,
        }
        losses = []
        accuracies = []
        lengths = []
        new_dataset = {}
        few_shot_dataset = self.policy_model.few_shot_dataset
        self.policy_model.few_shot_dataset = None
        dataloader = self.test_dataloader if test else self.eval_dataloader
        self.logger.info(f"Running evaluation on {len(dataloader.dataset)} samples")
        for i, batch in enumerate(tqdm(dataloader)):
            with torch.no_grad():
                ### Generate thoughts
                ids = batch['_id']
                prompts = batch['question']
                answers = batch['answer']
                datasets = batch['dataset'] if "dataset" in batch else None
                thoughts, responses = self.policy_model.run(prompts, log=True, datasets=datasets)
                datasets = datasets if datasets else [""] * len(prompts)
                for _id, prompt, thought, answer, response, dataset in zip(ids, prompts, thoughts, answers, responses, datasets):
                    new_dataset["_id"] = new_dataset.get("_id", []) + [_id]
                    new_dataset["question"] = new_dataset.get("question", []) + [prompt]
                    new_dataset["answer"] = new_dataset.get("answer", []) + [answer]
                    new_dataset["response"] = new_dataset.get("response", []) + [response]
                    new_dataset["thought"] = new_dataset.get("thought", []) + [thought]
                    new_dataset["dataset"] = new_dataset.get("dataset", []) + [dataset]
                    prompt = self.policy_model.format_prompts(questions=[prompt], datasets=[dataset])[0]
                    thought = self.thought_format.format(thought=thought) if thought else ""
                    response = self.assistant_format.format(assistant=response) + self.eos_token
                    prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).squeeze(0)
                    thought_ids = self.tokenizer.encode(thought, return_tensors="pt", add_special_tokens=False).squeeze(0)
                    response_ids = self.tokenizer.encode(response, return_tensors="pt", add_special_tokens=False).squeeze(0)
                    input_ids = torch.cat([prompt_ids, thought_ids, response_ids], dim=0).to(self.current_device).unsqueeze(0).long()
                    attention_mask = torch.cat([torch.ones_like(prompt_ids), torch.ones_like(thought_ids), torch.ones_like(response_ids)], dim=0).to(self.current_device).unsqueeze(0).long()
                    label = torch.cat([-100*torch.ones_like(prompt_ids), -100*torch.ones_like(thought_ids), response_ids], dim=0).to(self.current_device).unsqueeze(0).long()
                    with torch.no_grad():
                        losses.append(self.policy_model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=label).loss.item())
                    lengths.append(thought_ids.shape[-1])
                    accuracies.append(is_correct(response, answer))
                    new_dataset["length"] = new_dataset.get("length", []) + [thought_ids.shape[-1]]
                    new_dataset["loss"] = new_dataset.get("loss", []) + [losses[-1]]
                    new_dataset["accuracy"] = new_dataset.get("accuracy", []) + [accuracies[-1]]
        self.policy_model.few_shot_dataset = few_shot_dataset
        loss = torch.tensor(losses).float().mean().item()
        accuracy = torch.tensor(accuracies).float().mean().item()
        lengths_mean = torch.tensor(lengths).float().mean().item()
        lengths_std = torch.tensor(lengths).float().std().item()
        if test:
            result_df = pd.DataFrame(new_dataset)
            save_to(result_df, f"{self.output_path}/test_results.json")
            stats = {
                'test/step': step,
                'test/loss': loss,
                'test/average_length': lengths_mean,
                'test/std_length': lengths_std,
                'test/accuracy': accuracy,
            }
            stats_to_print = {k: f"{v:.3f}" for k, v in stats.items()}
            self.logger.info(f"Logging metrics: {stats_to_print}")
        else:
            result_df = pd.DataFrame(new_dataset)
            save_to(result_df, f"{self.output_path}/val_results.json")
            self.validation_losses[step] = loss
            self.validation_accs[step] = accuracy
                
    def save_checkpoint(self, step):
        '''Save checkpoint'''
        if step != len(self.train_dataloader)-1 and (step % self.args.save_steps != 0):
            return
        self.save()
        self.logger.info(f"Checkpoint saved at step {step}")
    

