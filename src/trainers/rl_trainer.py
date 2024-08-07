import logging
import torch
import warnings
import torch.nn.functional as F
import wandb
import os
import pandas as pd
import gc
from datetime import datetime
from hydra.utils import instantiate
from pathlib import Path
from dataclasses import dataclass, field
from datetime import date
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    DataCollator,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
    TrainingArguments
)
from reward_model import RewardModel
from policy_model import AgentPretrainedModel
from datasets import Dataset
from typing import Union, Dict, Optional, List, Any
from utils import (
    is_correct,
    save_to,
    trim,
    DATASET_TO_PREF
)
from torch.utils.tensorboard import SummaryWriter
import os
os.environ['CURL_CA_BUNDLE'] = ''

@dataclass
class RLTrainingArguments(TrainingArguments):

    batch_size: int = field(default=4, metadata={"help": "Batch size for data loader."})
    
    mini_batch_size: int = field(default=4, metadata={"help": "Mini batch size for rl training."})
    
    learning_rate: float = field(default=1e-4, metadata={"help": "Learning rate for optimizer."})
        
    epochs: int = field(default=3, metadata={"help": "Number of training epochs."})
    
    rl_epochs: int = field(default=1, metadata={"help": "Number of rl training epochs."})

    start_epoch: int = field(default=0, metadata={"help": "Epoch from which to start training."})
        
    log_with: str = field(default="tensorboard", metadata={"help": "Logging method."})
    
    output_dir: str = field(default=None, metadata={"help": "Output directory."})

    output_suffix: str = field(default="", metadata={"help": "Output suffix."})

    rollout_sample_size: int = field(default=4, metadata={"help": "Number of rollouts."})

    voc_gamma: float = field(default=5e-4, metadata={"help": "Value of computation gamma."})  

    logging_steps: int = field(default=10, metadata={"help": "Logging steps."}) 

    save_steps: int = field(default=10, metadata={"help": "Save steps."})

    eval_steps: int = field(default=10, metadata={"help": "Evaluation steps."})

    warmup_ratio: float = field(default=0.01, metadata={"help": "Warmup ratio."})

    max_grad_norm: float = field(default=0.5, metadata={"help": "Max gradient norm."})
    
#     gamma_advantage: float = field(default=1.0, metadata={"help": "Gamma advantage."})
    
#     lambda_advantage: float = field(default=0.95, metadata={"help": "Lambda advantage."})
    
#     cliprange: float = field(default=0.2, metadata={"help": "Cliprange."})
    
#     vf_coef: float = field(default=0.1, metadata={"help": "Value function coefficient."})
    
#     whiten_rewards: bool = field(default=False, metadata={"help": "Whiten rewards."}) 

#     celoss_coef: float = field(default=0.1, metadata={"help": "Cross entropy loss coefficient."})


class CustomRLDataCollator(DataCollatorForLanguageModeling):
        
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

class RLTrainer():

    def __init__(self,
            model: AgentPretrainedModel,
            args: RLTrainingArguments,
            train_dataset_path: str,
            ref_model: AgentPretrainedModel = None,
            eval_dataset_path: str = None,
            test_dataset_path: str = None,
            max_seq_length: Optional[int] = None,
            type: str = "",
            force_direct: bool = True
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
        self.ref_model = ref_model
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
        print(f"Datasets loaded: {len(self.train_dataset)} training samples, {len(self.eval_dataset)} evaluation samples, {len(self.test_dataset)} test samples")
        data_collator = CustomRLDataCollator(tokenizer=self.tokenizer, mlm=False)
        self.train_dataloader = self.prepare_dataloader(self.train_dataset, data_collator, batch_size=self.args.batch_size)
        self.eval_dataloader = self.prepare_dataloader(self.eval_dataset, data_collator, batch_size=self.args.mini_batch_size) if self.eval_dataset else None
        self.test_dataloader = self.prepare_dataloader(self.test_dataset, data_collator, batch_size=self.args.mini_batch_size) if self.test_dataset else None
        print(f"Dataloaders prepared: {len(self.train_dataloader)} training batches, {len(self.eval_dataloader)} evaluation batches, {len(self.test_dataloader)} test batches")
        self.data_collator = data_collator
        ### Optimizer and scheduler
        self.set_optimizer()
        date_str = date.today().strftime("%m%d")
        self.output_path = self.args.output_dir + self.policy_model.model_name.split("/")[-1] + "_" + date_str + self.args.output_suffix + "_epoch_0"
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
        self.reward_model.logger = logger
        self.policy_model.logger = logger

    def set_optimizer(self):
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.policy_model.parameters()), lr=self.args.learning_rate)
        self.total_steps = len(self.train_dataset) // self.args.mini_batch_size
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=int(self.args.warmup_ratio * self.total_steps), num_training_steps=self.total_steps)
        
    def reload_policy_model(self):
        del self.policy_model
        del self.reward_model
        gc.collect()
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
        self.policy_model.few_shot_dataset = None

    def load_dataset(self, path: str):
        '''Load dataset from path
        Args:
            path (str): Path to dataset
        Returns:
            dataset (Dataset): Dataset object
        '''
        try:
            df = pd.read_json(path)
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
            shuffle=True,
            drop_last=True
        )
        return dataloader

    def save(self):
        '''Save model'''
        self.policy_model.save_pretrained(self.output_path)
        self.tokenizer.save_pretrained(self.output_path)
        self.logger.info(f"Model saved at {self.output_path}")
    
    def rl_loss(self, step_data):
        '''Compute RL loss according to the chosen algorithm'''
        raise NotImplementedError
    
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
        targets_mask: torch.Tensor,
    ):  
        raise NotImplementedError

    def _format_data(self, data):
        # if len(self.tokenizer) == self.policy_model.pretrained_vocab_size and "<thought>" in self.policy_model.thought_format:
        #     self.logger.info("Adding special tokens")
        #     add_special_tokens(self.policy_model, self.tokenizer, self.ref_model)
        #     self.save_model()
        ### Prepare prompts, answers, thoughts and responses
        prompts = [trim(p) for p in data['question']]
        answers = [trim(c) for c in data['answer']]
        responses = [trim(a) for a in data['response']]
        thoughts = [trim(t) for t in data['thought']]
        instruction = self.policy_model.instruction
        dataset_prefixes = [DATASET_TO_PREF.get(d, "") for d in data['dataset']]
        prompts = [self.bos_token + self.user_format.format(user=instruction + d + p) for p, d in zip(prompts, dataset_prefixes)]
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

    def validation(self, step, test=False):
        '''Validation step'''
        if self.eval_dataloader is None:
            self.logger.info("No evaluation dataset provided")
            return
        self.logger.info(f"Step {step}: Validation | {step == 0}")
        if (step % self.args.eval_steps != 0) or (step == 0):
            return
        self.policy_model.generation_args = {
            "pad_token_id": self.tokenizer.eos_token_id,
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
                    prompt = self.bos_token + self.user_format.format(user=prompt)
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
                    new_dataset["_id"] = new_dataset.get("_id", []) + [_id]
                    new_dataset["question"] = new_dataset.get("question", []) + [prompt]
                    new_dataset["answer"] = new_dataset.get("answer", []) + [answer]
                    new_dataset["response"] = new_dataset.get("response", []) + [response]
                    new_dataset["thought"] = new_dataset.get("thought", []) + [thought]
                    new_dataset["dataset"] = new_dataset.get("dataset", []) + [dataset]
                    new_dataset["length"] = new_dataset.get("length", []) + [thought_ids.shape[-1]]
                    new_dataset["loss"] = new_dataset.get("loss", []) + [losses[-1]]
        if test:
            result_df = pd.DataFrame(new_dataset)
            save_to(result_df, f"{self.output_path}/results_{step}.json")
        self.policy_model.few_shot_dataset = few_shot_dataset
        loss = torch.tensor(losses).float().mean().item()
        accuracy = torch.tensor(accuracies).float().mean().item()
        lengths_mean = torch.tensor(lengths).float().mean().item()
        lengths_std = torch.tensor(lengths).float().std().item()
        if not test:
            self.validation_losses[step] = loss
            self.validation_accs[step] = accuracy
            if self.args.log_with == "wandb":
                stats = {
                    'eval/step': step,
                    'eval/loss': loss,
                    'eval/average_length': lengths_mean,
                    'eval/std_length': lengths_std,
                    'eval/accuracy': accuracy,
                }
                stats_to_print = {k: f"{v:.3f}" for k, v in stats.items()}
                self.logger.info(f"Logging metrics: {stats_to_print}")
                wandb.log(stats)
            elif self.args.log_with == "tensorboard":
                self.writer.add_scalar('eval/loss', loss, step)
        else:
            self.logger.info(f"Test loss: {loss}, accuracy: {accuracy}")
                
    def save_checkpoint(self, step):
        '''Save checkpoint'''
        self.save()
        if step % self.args.save_steps != 0 or step == 0:
            return
        self.checkpoint_losses[step] = self.validation_losses[step]
        self.checkpoint_accs[step] = self.validation_accs[step]
        if len(self.checkpoint_accs) > 1 and self.checkpoint_accs[step] < max(self.checkpoint_accs.values()):
            return
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        for f in os.listdir(self.args.output_dir):
            if f.startswith("checkpoint_"):
                os.remove(os.path.join(self.args.output_dir, f))
        torch.save({
            'policy_model': self.policy_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'step': step,
        }, f'{self.args.output_dir}/{"checkpoint_" + str(step)}.pth')
        self.logger.info(f'[reinforce_step {step}] model checkpoint saved')
    

