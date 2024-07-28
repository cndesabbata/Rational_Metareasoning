import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, Any, List, Union, Optional, Tuple
from dataclasses import dataclass, field
from transformers import (
    StoppingCriteria,
    StoppingCriteriaList,
    LlamaForCausalLM,
    PhiForCausalLM,
    GemmaForCausalLM,
    Phi3ForCausalLM,
    AutoModelForCausalLM,
    AutoTokenizer
)
from utils import (
    DATASET_TO_PREF,
    MODE_TO_INSTRUCTION,
    split_thought_and_answer,
    logprobs_from_logits
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss

@dataclass
class PolicyModelOutput(CausalLMOutputWithPast):
    logprobs: torch.FloatTensor = None
    value: Optional[torch.FloatTensor] = None

class AgentPretrainedModel:
    
    def __init__(self, config: Dict[str, Any]):
        model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, device_map="auto", torch_dtype="auto")
        self.model_name = config.model_name_or_path.split("/")[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.__dict__.update(model.__dict__)
        self.logger = None
        self.few_shot_dataset = pd.read_json(config.few_shot_path) if config.few_shot_path else None
        self.current_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.value_head = None 
        self.inference_mode = config.inference_mode
        self.instruction = MODE_TO_INSTRUCTION[config.inference_mode]
        self.generation_args = config.generation_args

    def add_meta_tokens(self):
        '''Use meta tokens to separate the thought from the answer.'''
        self.thought_format = "<thought>{thought}</thought>\n"

    def log_model_output(self, prompt, completion):
        '''Log the model output to the logger.'''
        if self.logger is not None:
            self.logger.info("*"*200)
            self.logger.info(f"HF QUERY")
            self.logger.info(prompt)
            self.logger.info("MODEL RESPONSE")
            self.logger.info(completion)
            self.logger.info("*"*200)

    def format_prompts(self, questions: List[str], completions: List[str] = None, datasets: List[str] = None, hint: bool = False):
        '''Formats the prompts for the model (with chat tokens, few-shot examples, etc.).
        Args:
            questions (List[str]): The list of questions to be formatted.
            completions (List[str], optional): The list of correct answers (used for hints). Defaults to None.
            datasets (List[str], optional): The list of datasets (to determine which few-shot examples to use). Defaults to None.
            hint (bool, optional): Whether to include hints in the prompts. Defaults to False.
        Returns:
            List[str]: The formatted prompts.
        '''
        ### If datasets are not provided, don't use few-shot examples
        prefixes = self.build_few_shot(datasets, hint=hint) if datasets is not None else ["" for _ in range(len(questions))]
        prompts = []
        if completions is None:
            completions = ["" for _ in range(len(questions))]
            hint = False
        datasets = datasets if datasets is not None else ["" for _ in range(len(questions))]
        ### Format the prompts
        for i, (q, c, pref, d) in enumerate(zip(questions, completions, prefixes, datasets)):
            hint_str = f"\n(The correct answer is {c})" if hint else ""
            data_pref = DATASET_TO_PREF.get(d, "")
            question = self.user_format.format(user=self.instruction + data_pref + q + hint_str)
            formatted_chat = self.tokenizer.bos_token + pref + question
            prompts.append(formatted_chat)
        return prompts

    def build_few_shot(self, datasets: List[str], hint: bool = False):
        '''Builds few-shot examples for the model.
        Args:
            datasets (List[str]): The list of datasets for which we need to sample few-shot examples.
            hint (bool, optional): Whether to include hints in the prompts. Defaults to False.
        Returns:
            List[str]: The few-shot examples.
        '''
        if self.few_shot_dataset is None:
            return ["" for _ in range(len(datasets))]
        prefixes = []
        for i, dataset in enumerate(datasets):
            use_hint = hint
            try:
                df = self.few_shot_dataset[self.few_shot_dataset["dataset"] == dataset].sample(5)
            except:
                prefixes.append("")
                continue
            questions = df['user'].tolist()
            thoughts = df['thought'].tolist()
            answers = df['answer'].tolist()
            data_pref = DATASET_TO_PREF.get(dataset, "")
            few_shots = []
            ### Format the few-shot examples as chat
            for i, (questions_i, thoughts_i, answers_i) in enumerate(zip(questions, thoughts, answers)):
                hint_str = f"\n(The correct answer is {answers_i})" if use_hint else ""
                question = self.user_format.format(user=self.instruction + data_pref + questions_i + hint_str)
                thought = self.thought_format.format(thought=thoughts_i) if self.inference_mode == "cot" else ""
                answer = self.assistant_format.format(assistant=answers_i)
                chat = question + "\n" + thought + answer + "\n"
                few_shots.append(chat)
            few_shots = "".join(few_shots) + ""
            prefixes.append(few_shots)
        return prefixes

    def run(self, input: Union[str, List[str]], log: bool = False, format: bool = True, datasets: List[str] = None):
        '''Run the model on the input.
        Args:
            input (Union[str, List[str]]): The user's input to the model (can be a single string or a list of strings).
            log (bool, optional): Whether to log the model output. Defaults to False.
            format (bool, optional): Whether to format the input. Defaults to True (input is not already formatted).
            datasets (List[str], optional): The datasets for which to use few-shot examples. Defaults to None.
        Returns:
            List[str]: The generated thoughts.
            List[str]: The generated answers.
        '''
        ### Prepare the input for generation
        all_thoughts = []
        all_answers = []
        if isinstance(input, str):
            input = [input]
        prompts = self.format_prompts(input, datasets=datasets) if format else input
        model_input = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.current_device)
        stopping_criteria = StoppingCriteriaList([
            AnswerStoppingCriteria(self.tokenizer, self.current_device),
        ])
        ### Generate the model output
        output_ids = self.generate(
            **model_input,
            **self.generation_args,
            stopping_criteria=stopping_criteria,
        )
        ### Decode and parse the model output
        outputs = self.tokenizer.batch_decode(output_ids[:, model_input['input_ids'].shape[1]:], skip_special_tokens=True)
        if log:
            for prompt, output in zip(prompts, self.tokenizer.batch_decode(output_ids[:, model_input['input_ids'].shape[1]:], skip_special_tokens=False)):
                self.log_model_output(prompt, output)
        all_thoughts, all_answers = split_thought_and_answer(outputs)
        return all_thoughts, all_answers

    def policy_forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            mask: torch.Tensor = None,
            compute_values: bool = False
            ) -> PolicyModelOutput:
        '''Forward pass of the model for RL training.
        Args:
            input_ids (torch.Tensor): The input token ids.
            attention_mask (torch.Tensor): The attention mask.
            mask (torch.Tensor, optional): The mask for the logprobs. Defaults to None.
            compute_values (bool, optional): Whether to compute the values for PPO. Defaults to False.
        Returns:
            PolicyModelOutput: The model output.
        '''
        input_ids = input_ids.to(self.current_device)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=True,
            return_dict=None,
        )
        hidden_states = outputs[0].to(self.current_device)
        logits = self.lm_head(hidden_states)
        logits = logits.float().to(self.current_device)
        logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:,1:], gather=True)
        mask = mask.float() if mask is not None else torch.ones_like(input_ids[:,1:]).float()
        logprobs = logprobs * mask
        if not compute_values:
            return PolicyModelOutput(
                value=None,
                logprobs=logprobs
            )
        if not self.value_head:
            self.value_head = nn.Linear(self.config.hidden_size, 1).to(self.current_device)
        values = self.value_head(hidden_states).squeeze(-1)[:,:-1]
        values = values * mask
        return PolicyModelOutput(
            value=values,
            logprobs=logprobs
        )

class AnswerStoppingCriteria(StoppingCriteria):
    '''Stops the generation when the model outputs the first newline character after the "Answer:" tokens.'''
    
    def __init__(self, tokenizer, device):
        self.stop_token = tokenizer.encode("\n", add_special_tokens=False)[-1]
        self.tokenizer = tokenizer
        self.answer_tokens = self.tokenizer.encode("\nAnswer:", add_special_tokens=False, return_tensors="pt").squeeze().to(device)[1:]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        batch_size = input_ids.shape[0]
        finished = torch.zeros(batch_size, dtype=torch.bool).to(input_ids.device).bool()
        contains_answer_tokens = (input_ids[:, -32:].unfold(1, len(self.answer_tokens), 1) == self.answer_tokens.unsqueeze(0)).all(-1).any(-1)
        ends_with_new_line = input_ids[:, -1] == self.stop_token
        ends_with_eos = input_ids[:, -1] == self.tokenizer.eos_token_id
        finished = ends_with_eos | (contains_answer_tokens & ends_with_new_line)
        return finished

class LlamaAgent(AgentPretrainedModel, LlamaForCausalLM):

    def __init__(self, config: Dict[str, Any]):
        AgentPretrainedModel.__init__(self, config)
        self.vocab_size = self.config.vocab_size
        self.pretrained_vocab_size = 32000

class PhiAgent(AgentPretrainedModel, PhiForCausalLM):

    def __init__(self, config: Dict[str, Any]):
        AgentPretrainedModel.__init__(self, config)
        self.vocab_size = self.config.vocab_size
        self.pretrained_vocab_size = 50295

class Phi3Agent(AgentPretrainedModel, Phi3ForCausalLM):

    def __init__(self, config: Dict[str, Any]):
        AgentPretrainedModel.__init__(self, config)
        self.vocab_size = self.config.vocab_size
        self.pretrained_vocab_size = 32064
        self.user_format = "<|user|>\n{user}\n<|end|>\n<|assistant|>"
        self.assistant_format = "Answer: {assistant}\n<|end|>"
        self.thought_format = "Thought: {thought}\n"
        
class GemmaAgent(AgentPretrainedModel, GemmaForCausalLM):

    def __init__(self, config: Dict[str, Any]):
        AgentPretrainedModel.__init__(self, config)
        self.vocab_size = self.config.vocab_size
        self.pretrained_vocab_size = 256000
        self.user_format = "User: {user}\n\n"
        self.assistant_format = "Answer: {assistant}\n"
        self.thought_format = "Thought: {thought}\n"