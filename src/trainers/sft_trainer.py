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
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
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
        self.data_collator = DataCollatorForCompletionOnlyLM(tokenizer=self.tokenizer, response_template = self.policy_model.response_template, instruction_template = self.policy_model.instruction_template)
        print(f"Dataloaders prepared: {len(self.train_dataloader)} training batches, {len(self.eval_dataloader)} evaluation batches, {len(self.test_dataloader)} test batches")
        self.mod_str = "_direct" if self.force_direct else ""
        self.output_path = self.args.output_dir + self.policy_model.model_name.split("/")[-1] + self.mod_str + self.args.output_suffix + "_step_0"

    def set_logger(self, logger):
        '''Set logger'''
        self.logger = logger
        self.policy_model.logger = logger
    
    def load_policy_model(self, step=0):
        self.logger.info(f"Loading model at step {step}")
        if step != 0:
            path = "_".join(self.output_path.split("_")[:-2]) + f"_step_{step-1}"
            self.model_config.config.model_name_or_path = path
        self.policy_model = instantiate(self.model_config)
        self.policy_model.logger = self.logger

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
            raise FileNotFoundError(f"Dataset not found at {path}")
        return dataset   

    def train(self):
        output_dir = self.output_path.split("models")+"outputs", 
        args = SFTConfig(
            output_dir = output_dir,
            overwrite_output_dir = True,
            do_train = True,
            do_eval = True,
            logging_steps = 4,
            eval_steps = 16,
            save_steps = 16,
            load_best_model_at_end=True,
            per_device_train_batch_size = self.args.mini_batch_size,
            per_device_eval_batch_size = self.args.mini_batch_size,
            logging_dir = output_dir,
            learning_rate = self.args.learning_rate,
            max_seq_length = self.policy_model.model_max_length,
            num_train_epochs = 1
        )
        trainer = SFTTrainer(
            model = self.policy_model,
            args = args,
            data_collator = self.data_collator,
            train_dataset = self.train_dataset,
            formatting_func = self.policy_model.formatting_func,
            processing_class = self.policy_model.tokenizer
        )
        trainer.train()
        self.save_checkpoint()
        # remove outputs
        os.remove(output_dir)
        
                
    def save_checkpoint(self):
        '''Save checkpoint'''
        self.policy_model.save_pretrained(self.output_path)
        self.tokenizer.save_pretrained(self.output_path)
        self.logger.info(f"Model saved at {self.output_path}")
    
