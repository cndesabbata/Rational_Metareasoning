import pandas as pd
import torch
from typing import Dict, Any, List, Union
from transformers import (
    LlamaForCausalLM,
    PhiForCausalLM,
    MistralForCausalLM,
    AutoModelForCausalLM,
    AutoTokenizer
)
from utils import (
    split_thought_and_answer,
    trim
)

class PretrainedModel:
    
    def __init__(self, config: Dict[str, Any]):
        ### Load model
        model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, device_map="auto", torch_dtype="auto")
        self.model_name = config.model_name_or_path.split("/")[-1]
        self.__dict__.update(model.__dict__)
        ### Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        ### Set attributes
        self.logger = None
        self.few_shot_dataset = pd.read_json(config.few_shot_path) if config.few_shot_path else None
        self.current_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.inference_mode = config.inference_mode
        self.generation_args = config.generation_args

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
        ### If the gold truth (completions) is not provided, hints must be disabled
        if completions is None:
            completions = ["" for _ in range(len(questions))]
            hint = False
        datasets = datasets if datasets is not None else ["" for _ in range(len(questions))]
        ### Format the prompts
        for i, (q, c, pref, d) in enumerate(zip(questions, completions, prefixes, datasets)):
            hint_str = f"\n(The correct answer is {c})" if hint else ""
            d = d.split("_")[0] if "mmlu" in d else d
            question = self.user_format.format(user=q + hint_str)
            suffix = "Answer:" if self.inference_mode == "direct" else ""
            formatted_chat = self.tokenizer.bos_token + pref + question + suffix
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
            few_shots = []
            ### Format the few-shot examples as chat
            current_length = 0
            for i, (questions_i, thoughts_i, answers_i) in enumerate(zip(questions, thoughts, answers)):
                hint_str = f"\n(The correct answer is {answers_i})" if use_hint else ""
                question = self.user_format.format(user=trim(questions_i) + hint_str)
                thought = self.thought_format.format(thought=trim(thoughts_i)) if (self.inference_mode != "direct" and thoughts_i != "") else ""
                answer = self.assistant_format.format(assistant=trim(answers_i))
                chat = question + thought + answer
                current_length += len(self.tokenizer.encode(chat, return_tensors="pt")[0])
                few_shots.append(chat)
                if current_length > self.model_max_length // 2:
                    break
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
        ### Generate the model output
        output_ids = self.generate(
            **model_input,
            **self.generation_args
        )
        ### Decode and parse the model output
        outputs = self.tokenizer.batch_decode(output_ids[:, model_input['input_ids'].shape[1]:], skip_special_tokens=True)
        if log:
            for prompt, output in zip(prompts, self.tokenizer.batch_decode(output_ids[:, model_input['input_ids'].shape[1]:], skip_special_tokens=False)):
                self.log_model_output(prompt, output)
        if self.inference_mode == "direct":
            all_thoughts, all_answers = [""]*len(input), [o.split("\n")[0] for o in outputs]
        else:
            all_thoughts, all_answers = split_thought_and_answer(outputs)
        return all_thoughts, all_answers

class LlamaModel(PretrainedModel, LlamaForCausalLM):

    def __init__(self, config: Dict[str, Any]):
        PretrainedModel.__init__(self, config)
        self.user_format = "Question: {user}\n"
        self.assistant_format = "Answer: {assistant}\n\n"
        self.thought_format = "Thought: {thought}\n"
        self.stop_token = "\n\n"
        self.model_max_length = 8192

class MistralModel(PretrainedModel, MistralForCausalLM):

    def __init__(self, config: Dict[str, Any]):
        PretrainedModel.__init__(self, config)
        self.user_format = "Question: {user}\n"
        self.assistant_format = "Answer: {assistant}\n\n"
        self.thought_format = "Thought: {thought}\n"
        self.stop_token = "\n\n"
        self.model_max_length = 8192

class Phi2Model(PretrainedModel, PhiForCausalLM):

    def __init__(self, config: Dict[str, Any]):
        PretrainedModel.__init__(self, config)
        self.user_format = "Question: {user}\n"
        self.assistant_format = "Answer: {assistant}\n\n"
        self.thought_format = "Thought: {thought}\n"
        self.stop_token = "\n\n"
        self.model_max_length = 2048