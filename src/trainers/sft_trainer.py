import torch
import torch.nn.functional as F
import wandb
import os
import pandas as pd
from datetime import datetime
from torch.nn.utils import clip_grad_norm_
from hydra.utils import instantiate
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
    TrainingArguments
)
from policy_model import PretrainedModel
from datasets import Dataset
from typing import Union, Dict, Optional, List, Any
from torch.utils.tensorboard import SummaryWriter
import os
os.environ['CURL_CA_BUNDLE'] = ''

@dataclass
class SFTTrainingArguments(TrainingArguments):

    batch_size: int = field(default=4, metadata={"help": "Batch size for data loader."})
    
    learning_rate: float = field(default=1e-4, metadata={"help": "Learning rate for optimizer."})
        
    log_with: str = field(default="tensorboard", metadata={"help": "Logging method."})
    
    output_dir: str = field(default=None, metadata={"help": "Output directory."})

    output_suffix: str = field(default="", metadata={"help": "Output suffix."})

    logging_steps: int = field(default=10, metadata={"help": "Logging steps."}) 

    save_steps: int = field(default=10, metadata={"help": "Save steps."})

    eval_steps: int = field(default=10, metadata={"help": "Evaluation steps."})

    warmup_ratio: float = field(default=0.01, metadata={"help": "Warmup ratio."})

    max_grad_norm: float = field(default=0.5, metadata={"help": "Max gradient norm."})

class SFTDataCollator(DataCollatorForLanguageModeling):

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch_longest = max([len(e["input_ids"]) for e in examples])
        batch = {
            "input_ids": torch.stack([ F.pad(torch.tensor(e["input_ids"]), (batch_longest - len(e["input_ids"]), 0), value = self.tokenizer.pad_token_id) for e in examples ]),
            "attention_mask": torch.stack([ F.pad(torch.tensor(e["attention_mask"]), (batch_longest - len(e["attention_mask"]), 0), value = 0) for e in examples ]),
            "labels": torch.stack([ F.pad(torch.tensor(e["labels"]), (batch_longest - len(e["labels"]), 0), value = self.tokenizer.pad_token_id) for e in examples ])
            }
        return batch

class SFTTrainer():

    def __init__(self,
            model: PretrainedModel,
            args: SFTTrainingArguments,
            train_dataset_path: str,
            eval_dataset_path: str = None,
            test_dataset_path: str = None,
            max_seq_length: Optional[int] = None,
            type: str = "",
        ):
        self.logger = None
        ### Training arguments
        self.args = args
        self.current_device = model.device
        self.tokenizer = model.tokenizer
        self.max_seq_length = max_seq_length or self.tokenizer.model_max_length
        self.training_losses = {}
        self.validation_losses = {}
        self.test_losses = {}
        self.type = type
        ### Models
        self.policy_model = model
        self.model_config = None
        self.thought_format = self.policy_model.thought_format
        self.user_format = self.policy_model.user_format
        self.assistant_format = self.policy_model.assistant_format
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        ### Dataloaders
        self.train_dataset = self.load_dataset(train_dataset_path)
        self.eval_dataset = self.load_dataset(eval_dataset_path) if eval_dataset_path else None
        self.test_dataset = self.load_dataset(test_dataset_path) if test_dataset_path else None
        print(f"Datasets loaded: {len(self.train_dataset)} training samples, {len(self.eval_dataset)} evaluation samples, {len(self.test_dataset)} test samples")
        data_collator = SFTDataCollator(tokenizer=self.tokenizer, mlm=False)
        self.train_dataloader = self.prepare_dataloader(self.train_dataset, data_collator, batch_size=self.args.batch_size)
        self.eval_dataloader = self.prepare_dataloader(self.eval_dataset, data_collator, batch_size=self.args.batch_size) if self.eval_dataset else None
        self.test_dataloader = self.prepare_dataloader(self.test_dataset, data_collator, batch_size=self.args.batch_size) if self.test_dataset else None
        print(f"Dataloaders prepared: {len(self.train_dataloader)} training batches, {len(self.eval_dataloader)} evaluation batches, {len(self.test_dataloader)} test batches")
        self.data_collator = data_collator
        ### Optimizer and scheduler
        self.set_optimizer()
        # date_str = "_" + date.today().strftime("%m%d")
        date_str = ""
        self.output_path = self.args.output_dir + self.policy_model.model_name + date_str + self.args.output_suffix
        if self.args.log_with == "wandb":
            run_name = datetime.now().strftime("%m%d_%H%M%S")
            wandb_dir = self.args.output_dir
            if not os.path.exists(wandb_dir):
                os.makedirs(wandb_dir)
            wandb.init(project='metareasoning', name=run_name, dir=args.output_dir, mode="offline")
            wandb.define_metric('train/step')
            wandb.define_metric('eval/step')
            wandb.define_metric('train/*', step_metric='train/step')
            wandb.define_metric('eval/*', step_metric='eval/step', summary='max')
        elif self.args.log_with == "tensorboard":
            if not os.path.exists(self.args.output_dir):
                os.makedirs(self.args.output_dir)
            if not os.path.exists(self.args.output_dir + "/runs"): 
                os.makedirs(self.args.output_dir + "/runs")
            else:
                os.system(f"rm -fr {self.args.output_dir}/runs/*")
            self.writer = SummaryWriter(log_dir=self.args.output_dir + "/runs")

    def set_logger(self, logger):
        '''Set logger'''
        self.logger = logger
        self.policy_model.logger = logger

    def set_optimizer(self):
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.policy_model.parameters()), lr=self.args.learning_rate)
        self.total_steps = len(self.train_dataset) // self.args.batch_size
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=int(self.args.warmup_ratio * self.total_steps), num_training_steps=self.total_steps)
        
    def load_policy_model(self, step=0):
        self.logger.info(f"Loading model at step {step}")
        if step != 0:
            path = "_".join(self.output_path.split("_")[:-2]) + f"_step_{step-1}"
            self.model_config.config.model_name_or_path = path
        self.policy_model = instantiate(self.model_config)
        self.policy_model.logger = self.logger
        self.set_optimizer()
        # if step != 0:
        #     self.policy_model.few_shot_dataset = None

    def load_dataset(self, path: str, shuffle=True):
        '''Load dataset from path
        Args:
            path (str): Path to dataset
        Returns:
            dataset (Dataset): Dataset object
        '''
        def _tokenize(row: pd.Series):
            question = row['question']
            answer = row['answer']
            prompt = self.bos_token + self.user_format.format(user=question)
            completion = self.assistant_format.format(assistant=answer) + self.eos_token
            if self.policy_model.inference_mode != "direct" and "thought" in row and row["thought"] != "":
                thought = row["thought"]
                completion = self.thought_format.format(thought=thought) + completion
            ### Tokenize prompt
            prompt_input = self.tokenizer(prompt)
            prompt_input_ids = prompt_input["input_ids"]
            prompt_attention_mask = prompt_input["attention_mask"]
            ### Tokenize completion
            completion_input = self.tokenizer(completion, add_special_tokens=False)
            completion_input_ids = completion_input["input_ids"]
            completion_attention_mask = completion_input["attention_mask"]
            ### Create input
            input_ids = torch.tensor(prompt_input_ids + completion_input_ids)
            attention_mask = torch.tensor(prompt_attention_mask + completion_attention_mask)
            labels = torch.tensor([-100] * len(prompt_input_ids) + completion_input_ids)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        try:
            df = pd.read_json(path)
            if shuffle:
                df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            dataset = Dataset.from_pandas(df, preserve_index=False)
            dataset = dataset.map(_tokenize)
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
    
    def cross_entropy_loss(self, input_ids, attention_mask, labels):
        '''Compute cross entropy loss
        Args:
            input_ids (torch.Tensor): tokenized input 
            attention_mask (torch.Tensor): attention mask
            targets_mask (torch.Tensor): mask to identify labels
        Returns:
            loss (torch.Tensor): Cross entropy loss
        '''
        outputs = self.policy_model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss

    def train(self):
        for step, batch in enumerate(tqdm(self.train_dataloader)):
            ### Validate
            self.validation(step)
            ### Save checkpoint
            self.save_checkpoint(step)
            ### Train
            self.policy_model.train()
            self.optimizer.zero_grad()
            input_ids = batch["input_ids"].to(self.current_device)
            attention_mask = batch["attention_mask"].to(self.current_device)
            labels = batch["labels"].to(self.current_device)
            loss = self.cross_entropy_loss(input_ids, attention_mask, labels)
            loss.backward()
            clip_grad_norm_(self.policy_model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.training_losses[step] = loss.item()
            ### Log
            if step % self.args.logging_steps == 0:
                self.logger.info(f"Step: {step}, Loss: {loss.item()}")
                stats = {
                    'train/step': step,
                    'train/loss': loss.item()
                }
                if self.args.log_with == "wandb":
                    wandb.log(stats)
                elif self.args.log_with == "tensorboard":
                    self.writer.add_scalar('train/loss', loss, step)

    def validation(self, step, test=False):
        '''Validation step'''
        if step % self.args.eval_steps != 0:
            return
        self.policy_model.eval()
        dataloader = self.test_dataloader if test else self.eval_dataloader
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.current_device)
            attention_mask = batch["attention_mask"].to(self.current_device)
            labels = batch["labels"].to(self.current_device)
            loss = self.cross_entropy_loss(input_ids, attention_mask, labels)
            total_loss += loss.item()
        total_loss /= len(dataloader)
        if not test:
            self.logger.info(f"Validation loss: {total_loss}")
            self.validation_losses[step] = total_loss
            stats = {
                    'eval/step': step,
                    'eval/loss': total_loss
                }
            if self.args.log_with == "wandb":
                wandb.log(stats)
            elif self.args.log_with == "tensorboard":
                 self.writer.add_scalar('eval/loss', loss, step)                            
        else:
            self.logger.info(f"Test loss: {total_loss}")
            self.test_losses[step] = total_loss
            stats = {
                    'test/step': step,
                    'test/loss': total_loss
                }
            if self.args.log_with == "wandb":
                wandb.log(stats)
            elif self.args.log_with == "tensorboard":
                 self.writer.add_scalar('test/loss', loss, step)
        self.logger.info(f"Logging stats: {stats}")
                
    def save_checkpoint(self, step):
        '''Save checkpoint'''
        if step % self.args.save_steps != 0:
            return
        if len(self.validation_losses) != 0 and self.validation_losses[step] != min(self.validation_losses.values()):
            return
        self.policy_model.save_pretrained(self.output_path)
        self.tokenizer.save_pretrained(self.output_path)
        self.logger.info(f"Model saved at {self.output_path}")
        self.logger.info(f"Checkpoint saved at step {step}")
    
