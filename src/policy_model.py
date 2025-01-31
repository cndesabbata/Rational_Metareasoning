import pandas as pd
import torch
from typing import Dict, Any, List, Union
from transformers import (
    LlamaForCausalLM,
    PhiForCausalLM,
    MistralForCausalLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList
)
from utils import (
    split_thought_and_answer,
    trim,
    parse_model_output
)
import re
from trl import DataCollatorForCompletionOnlyLM

INSTRUCTION = "Answer the following question, thinking step by step to get to the answer. You can think however long you need, but answer as soon as you're ready. Keep you response concise and use the minimum number of steps to get to the answer. Once you're finished thinking, write your answer after the 'Answer: ' prompt.\n\n"

class PretrainedModel:
    
    def __init__(self, config: Dict[str, Any]):
        ### Load model
        model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, device_map="auto", torch_dtype="auto")
        self.model_name = config.model_name_or_path.split("/")[-1]
        self.model_path = config.model_name_or_path
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
        self.few_shots = self.build_few_shots()
        self.current_device = model.device
        self.inference_mode = config.inference_mode
        self.generation_args = config.generation_args
        self.instruction = INSTRUCTION if config.instruction else ""

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
        def _get_few_shots(d: str):
            if hint:
                d += "_hint"
            if d in self.few_shots:
                return self.few_shots[d]
            return self.few_shots["default"]
        prefixes = [_get_few_shots(d) for d in datasets] if datasets is not None else ["" for _ in range(len(questions))]
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
            question = self.user_format.format(user=self.instruction + q + hint_str)
            suffix = "Answer:" if self.inference_mode == "direct" else "Thought:"
            formatted_chat = self.tokenizer.bos_token + pref + question + suffix
            prompts.append(formatted_chat)
        return prompts
    
    def format_completions(self, answers: List[str], thoughts: List[str] = None):
        '''Formats the completions for the model (with chat tokens, few-shot examples, etc.).
        Args:
            answers (List[str]): The list of answers to be formatted.
            thoughts (List[str], optional): The list of thoughts to be formatted. Defaults to None.
        Returns:
            List[str]: The formatted completions.
        '''
        completions = []
        if thoughts is None:
            thoughts = ["" for _ in range(len(answers))]
        for a, t in zip(answers, thoughts):
            thought = self.thought_format.format(thought=t) if t != "" else ""
            completion = self.assistant_format.format(assistant=thought + a)
            completions.append(completion)
        return completions
    
    def formatting_func(self, example):
        return [f"{self.format_prompts([e['question']])[0]}{self.format_completions([e['answer']], [e['thought']])[0]}" for e in example]

    def build_few_shots(self):
        '''Builds few-shot examples for the model.
        Args:
            datasets (List[str]): The list of datasets for which we need to sample few-shot examples.
            hint (bool, optional): Whether to include hints in the prompts. Defaults to False.
        Returns:
            List[str]: The few-shot examples.
        '''
        ### Set empty string as default value
        few_shots_dict = dict()
        if self.few_shot_dataset is None:
            return few_shots_dict
        for dataset in self.few_shot_dataset["dataset"].unique():
            for hint in [True, False]:
                df = self.few_shot_dataset[self.few_shot_dataset["dataset"] == dataset].sample(5)
                questions = df['user'].tolist()
                thoughts = df['thought'].tolist()
                answers = df['answer'].tolist()
                few_shots = []
                ### Format the few-shot examples as chat
                for i, (questions_i, thoughts_i, answers_i) in enumerate(zip(questions, thoughts, answers)):
                    hint_str = f"\n(The correct answer is {answers_i})" if hint else ""
                    question = self.user_format.format(user=questions_i + hint_str)
                    thought = self.thought_format.format(thought=thoughts_i) if thoughts_i != "" else ""
                    answer = self.assistant_format.format(assistant= thoughts_i + "Answer: " + answers_i)
                    chat = question + answer
                    few_shots.append(chat)
                hint_key = "_hint" if hint else ""
                few_shots_dict[f"{dataset}{hint_key}"] = "\n\n".join(few_shots)
        ### Add default option for few-shot examples
        # Sample one line per dataset
        default = self.few_shot_dataset.sample(frac=1, random_state=1).groupby("dataset", sort=False).first().reset_index()
        questions = default['user'].tolist()
        thoughts = default['thought'].tolist()
        answers = default['answer'].tolist()
        few_shots = []
        for i, (questions_i, thoughts_i, answers_i) in enumerate(zip(questions, thoughts, answers)):
            question = self.user_format.format(user=questions_i)
            thought = self.thought_format.format(thought=thoughts_i) if thoughts_i != "" else ""
            answer = self.assistant_format.format(assistant=answers_i)
            chat = question + thought + answer
            few_shots.append(chat)
        few_shots_dict["default"] = "\n\n".join(few_shots)
        return few_shots_dict
    
    def parse_model_output(self, outputs: List[str]) -> tuple:
        thoughts, answers = [], []
        for output in outputs:
            try:
                # Answer is part between "Answer:" and the next newline character. If more exist, it's the last one.
                answer = re.findall(r"Answer:(.*?)\n", output, re.DOTALL)[-1].strip()
            except:
                answer = ""
            try:
                # Thought is part between thought tags. If more exist, it's the last one.
                thought = re.findall(r"<thought>(.*?)</thought>", output, re.DOTALL)[-1].strip()
            except:
                thought = ""
            answers.append(answer)
            thoughts.append(thought)
        return thoughts, answers

    def run(self, prompts: List[str], log: bool = False):
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
        model_input = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.current_device)
        stopping_criteria = StoppingCriteriaList([
            AnswerStoppingCriteria(self.tokenizer, self.current_device),
        ])
        ### Generate the model output
        output_ids = self.generate(
            **model_input,
            **self.generation_args,
            stopping_criteria=stopping_criteria
        )
        ### Decode and parse the model output
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        if log:
            for prompt, output in zip(prompts, self.tokenizer.batch_decode(output_ids[:, model_input['input_ids'].shape[1]:], skip_special_tokens=False)):
                self.log_model_output(prompt, output)
        all_thoughts, all_answers = self.parse_model_output(outputs)
        return all_thoughts, all_answers
    
class AnswerStoppingCriteria(StoppingCriteria):
    '''Stops the generation when the model outputs the first newline character after the "Answer:" tokens.'''

    def __init__(self, tokenizer, device):
        self.stop_tokens = self.user_template
        self.answer_tokens = "Answer:"
        self.tokenizer = tokenizer
        self.prefix_length = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        batch_size = input_ids.shape[0]
        if self.prefix_length is None:
            self.prefix_length = input_ids.shape[1] - 1
        finished = torch.zeros(batch_size, dtype=torch.bool).to(input_ids.device).bool()
        ### Check once every 16 tokens
        if input_ids.shape[1] % 16 != 0:
            return finished
        detokenized_generations = self.tokenizer.batch_decode(input_ids[:, self.prefix_length:], skip_special_tokens=True)
        for i, end in enumerate(detokenized_generations):
            if self.stop_tokens in end and self.answer_tokens in end:
                finished[i] = True
        return finished

class LlamaModel(PretrainedModel, LlamaForCausalLM):

    def __init__(self, config: Dict[str, Any]):
        PretrainedModel.__init__(self, config)
        self.user_format = "[User] {user}\n"
        self.assistant_format = "[Assistant] {assistant}\n"
        self.response_template = "[Assistant]"
        self.user_template = "[User]"
        self.thought_format = "<thought> {thought} </thought>\n"
        self.model_max_length = 2048

class LlamaInstructModel(PretrainedModel, LlamaForCausalLM):

    def __init__(self, config: Dict[str, Any]):
        PretrainedModel.__init__(self, config)
        self.user_format = "<|start_header_id|>user<|end_header_id|>\n{user}\n<|eot_id|>"
        self.assistant_format = "<|start_header_id|>assistant<|end_header_id|>\n{assistant}\n<|eot_id|>"
        self.instruction_template = "<|start_header_id|>user<|end_header_id|>"
        self.response_template = "<|start_header_id|>assistant<|end_header_id|>"
        self.thought_format = "<thought> {thought} </thought>\n"
        self.model_max_length = 2048

class MistralModel(PretrainedModel, MistralForCausalLM):

    def __init__(self, config: Dict[str, Any]):
        PretrainedModel.__init__(self, config)
        self.user_format = "[User] {user}\n"
        self.assistant_format = "[Assistant] {assistant}\n"
        self.response_template = "[Assistant]"
        self.user_template = "[User]"
        self.thought_format = "<thought> {thought} </thought>\n"
        self.model_max_length = 2048
        
class MistralInstructModel(PretrainedModel, MistralForCausalLM):

    def __init__(self, config: Dict[str, Any]):
        PretrainedModel.__init__(self, config)
        self.user_format = "[INST]\n{user}\n[/INST]\n"
        self.assistant_format = "{assistant}\n"
        self.instruction_template = "[INST]"
        self.response_template = "[/INST]"
        self.thought_format = "<thought> {thought} </thought>\n"
        self.stop_token = "\n\n"
        self.model_max_length = 2048

class Phi2Model(PretrainedModel, PhiForCausalLM):

    def __init__(self, config: Dict[str, Any]):
        PretrainedModel.__init__(self, config)
        self.user_format = "[User] {user}\n"
        self.assistant_format = "[Assistant] {assistant}\n"
        self.response_template = "[Assistant]"
        self.user_template = "[User]"
        self.thought_format = "<thought> {thought} </thought>\n"
        self.model_max_length = 2048