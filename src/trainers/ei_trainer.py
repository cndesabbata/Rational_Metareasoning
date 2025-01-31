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
    trim,
    print_gpu_utilization
)
import os
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
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
        self.base_model_name = model.model_path
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
        self.data_collator = CustomEIDataCollator(tokenizer=self.tokenizer, mlm=False)
        self.completion_collator = DataCollatorForCompletionOnlyLM(tokenizer=self.tokenizer, response_template = self.policy_model.response_template, instruction_template = self.policy_model.instruction_template)
        self.train_dataloader = self.prepare_dataloader(self.train_dataset, self.data_collator, batch_size=self.args.batch_size)
        self.eval_dataloader = self.prepare_dataloader(self.eval_dataset, self.data_collator, batch_size=self.args.mini_batch_size) if self.eval_dataset else None
        self.test_dataloader = self.prepare_dataloader(self.test_dataset, self.data_collator, batch_size=self.args.mini_batch_size) if self.test_dataset else None
        self.mod_str = "_direct" if self.force_direct else ""
        self.output_path = self.args.output_dir + self.policy_model.model_name.split("/")[-1] + self.mod_str + self.args.output_suffix + "_step_0"


    def set_logger(self, logger):
        '''Set logger'''
        self.logger = logger
        self.reward_model.logger = logger
        self.policy_model.logger = logger
       
    def load_policy_model(self, step=-1):
        '''Load policy model at a specific step. Can be used to load a checkpoint.'''
        self.logger.info(f"Loading model at step {step}")
        if step >= 0:
            self.model_config.config.model_name_or_path = "_".join(self.output_path.split("_")[:-2]) + f"_step_{step}"
        else:
            self.model_config.config.model_name_or_path = self.base_model_name
        self.policy_model = instantiate(self.model_config)
        self.policy_model.logger = self.logger
        self.reward_model = RewardModel(
            model=self.policy_model,
            voc_gamma=self.args.voc_gamma,
            logger=self.logger,
            sample_size=self.args.rollout_sample_size,
            device=self.current_device,
            tokenizer=self.tokenizer)
        self.policy_model.model_name = self.policy_model.model_name.split("/")[-1].split("_")[0]

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

    def save(self):
        '''Save model'''
        self.policy_model.save_pretrained(self.output_path)
        self.tokenizer.save_pretrained(self.output_path)
        self.logger.info(f"Model saved at {self.output_path}")
        current_step = int(self.output_path.split("_")[-1])
        if current_step > 0:
            for i in range(current_step):
                previous_path = "_".join(self.output_path.split("_")[:-2]) + f"_step_{i}"
                for file in os.listdir(previous_path):
                    if file == "config.json" or file == "val_results.json":
                        continue
                    os.remove(os.path.join(previous_path, file))

    # def step(
    #     self,
    #     input_ids: torch.Tensor,
    #     attention_mask: torch.Tensor,
    #     thoughts_mask: torch.Tensor,
    #     targets_mask: torch.Tensor
    # ):  
    #     ### Select best input_ids, based on rewards
    #     thoughts_mask = torch.clamp(thoughts_mask + targets_mask, 0, 1)
    #     labels = torch.where(thoughts_mask == 1, input_ids, torch.tensor(-100).to(self.current_device))
    #     ### Prepare step data
    #     step_data = {
    #         "input_ids": input_ids,
    #         "attention_mask": attention_mask,
    #         "labels": labels,
    #     }
    #     ### Compute loss
    #     self.policy_model.train()
    #     train_losses = []
    #     bs = input_ids.shape[0]
    #     step_size = self.args.mini_batch_size
    #     ### Clear cache
    #     self.clear_cache()
    #     for _, mini_step_start in enumerate(range(0, bs, step_size)):
    #         mini_step_end = mini_step_start + step_size
    #         self.logger.info(f"Selecting mini-batch: {mini_step_start}:{mini_step_end}")
    #         mini_batch = {k: v[mini_step_start:mini_step_end] for k, v in step_data.items()}
    #         loss_total = self.rl_loss(mini_batch)
    #         train_losses.append(loss_total.detach().item())
    #         loss_total.backward()
    #         self.optimizer.step()
    #         self.scheduler.step()
    #         ### Clear cache
    #         del loss_total
    #         self.optimizer.zero_grad()

    #     return np.mean(train_losses)
    
    def training_step(self, batch_data: Dict[str, Any]):
        args = SFTConfig(
            output_dir = None, 
            do_train = True,
            eval_strategy = "no",
            bf16 = self.policy_model.model.config.torch_dtype == "bfloat16",
            per_device_train_batch_size = self.args.mini_batch_size,
            learning_rate = self.args.learning_rate,
            max_seq_length = self.policy_model.model_max_length,
            num_train_epochs = 1
        )
        trainer = SFTTrainer(
            model = self.policy_model,
            args = args,
            data_collator = self.completion_collator,
            train_dataset = Dataset.from_dict(batch_data),
            formatting_func = self.policy_model.formatting_func,
            processing_class = self.policy_model.tokenizer
        )
        trainer.train()
        

    def sample_rollouts(self, batch_data: Dict[str, str], hint: bool = False):
        raise NotImplementedError

    def generate_data(self, step: int = 0, batch: Optional[Dict[str, str]] = None) -> Dict:
        columns = ["_id", "question", "answer", "thought", "response", "dataset", "reward"]
        # Construct path to the dataset file.
        new_path = os.path.join(
            self.args.data_dir,
            f"train_{self.policy_model.model_name}{self.mod_str}{self.args.output_suffix}_step_{step}.json"
        )
        # Attempt to load the dataset if it already exists.
        try:
            dataset = self.load_dataset(new_path, shuffle=False)
            return dataset.to_dict()
        except FileNotFoundError:
            pass
        # If the file does not exist, sample rollouts from the policy model.
        self.logger.info("Sampling rollouts from policy model")
        use_hint = self.args.use_hint
        new_batch = self.sample_rollouts(batch, hint=use_hint, step=step).to_dict()
        new_dataset = Dataset.from_dict(new_batch)
        # Convert to a pandas DataFrame, reorder columns, shuffle, and reset index.
        df = new_dataset.to_pandas()[columns].sample(frac=1, random_state=42).reset_index(drop=True)
        # Ensure the output directory exists and save the new dataset.
        os.makedirs(self.args.data_dir, exist_ok=True)
        save_to(df, new_path)
        return new_dataset.to_dict()

    def train(self):
        self.logger.info(f"Starting training: {self.args.start_step}/{len(self.train_dataloader)}")
        previous_batch = None
        for step, batch in enumerate(tqdm(self.train_dataloader)):
            self.logger.info(f"Running step {step}/{len(self.train_dataloader)}")
            print_gpu_utilization()
            previous_batch = {k: v + previous_batch[k] for k, v in batch.items()} if previous_batch is not None else batch
            if self.args.start_step > step:
                continue
            self.policy_model.eval()
            if step == self.args.start_step:
                self.load_policy_model(step-1)
            new_data = self.generate_data(step, previous_batch)
            batch = new_data
            ### Reset model
            self.load_policy_model(step=-1)
            self.output_path = "_".join(self.output_path.split("_")[:-2]) + f"_step_{step}"
            self.policy_model.train()
            ### Train model
            self.training_step(batch)
            ### Validation and Saving
            self.save_checkpoint(step)
            self.validation(step)
        self.validation(step, test=True)

    def validation(self, step, test=False):
        '''Validation step'''
        if self.eval_dataloader is None and not test:
            self.logger.info("No evaluation dataset provided")
            return
        self.logger.info(f"Step {step}: Validation | Test {test}")
        if step % self.args.eval_steps != 0 and not test:
            return
        ### Save training args and temporarily replace them
        training_generation_args = self.policy_model.generation_args
        training_few_shot_dataset = self.policy_model.few_shot_dataset
        eval_generation_args = {
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": False,
            "max_new_tokens": 512,
        }
        self.policy_model.generation_args = eval_generation_args
        self.policy_model.few_shot_dataset = None
        ### Run evaluation
        accuracies = []
        lengths = []
        new_df = []
        dataloader = self.test_dataloader if test else self.eval_dataloader
        self.logger.info(f"Running evaluation on {len(dataloader.dataset)} samples")
        for i, batch in enumerate(tqdm(dataloader)):
            prompts = self.policy_model.format_prompts(batch['question'])
            thoughts, responses = self.policy_model.run(prompts, log=True)
            correct = [is_correct(response, answer) for response, answer in zip(responses, batch['answer'])]
            lengths += [len(self.tokenizer.encode(thought)) for thought in thoughts]
            accuracies += correct
            new_df += [pd.DataFrame({
                "_id": batch['_id'],
                "question": batch['question'],
                "answer": batch['answer'],
                "thought": thoughts,
                "response": responses,
                "correct": correct
            })]
        new_df = pd.concat(new_df)
        self.policy_model.generation_args = training_generation_args
        self.policy_model.few_shot_dataset = training_few_shot_dataset
        stats = {"step": step, "accuracy": f"{np.mean(accuracies)*100:.2f}%", "length": f"{np.mean(lengths):.2f}"}
        self.logger.info(f"Validation{(' (Test)' if test else '')} stats: {stats}")
        if test:
            save_to(new_df, f"{self.output_path}/test_results.json")
        else:
            save_to(new_df, f"{self.output_path}/val_results.json")
                
    def save_checkpoint(self, step):
        '''Save checkpoint'''
        if step != len(self.train_dataloader)-1 and (step % self.args.save_steps != 0):
            return
        self.save()
        self.logger.info(f"Checkpoint saved at step {step}")
    

