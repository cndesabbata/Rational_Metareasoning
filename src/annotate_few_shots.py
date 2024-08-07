
import pandas as pd
import argparse
import yaml
import tiktoken
import os
import time
from typing import Dict, List
from openai import OpenAI
from datetime import date
from utils import set_logger, save_to
from typing import Union
from tqdm import tqdm

SYSTEM_PROMPT = """You are an intelligent and helpful assistant, who has to solve the user's problems. The user will ask you a question and you will have to provide the best possible answer. You can think step by step to get to the solution. You must follow these instructions:
- Read the question carefully.
- You MUST use the minimum number of steps to get to the solution.
- You can use structured reasoning and math/logic symbols if necessary, but write it in plain text (don't use LaTeX).
- Separate thought ("Thought: ...") and answer ("Answer: ...")
- When reasoning, you can use first-person singular pronouns.
- The user could give you a hint to help you solve the problem. If this is the case, you must not mention the hint in your answer.
Remember to be concise in your reasoning and to provide a clear and simple answer to the user's question.
"""

USER_PROMPT = """Solve the following problem.\n\n{question}\n{hint}\n"""

HINT_STRING = "[Correct answer: {answer}]"


COSTS_PER_TOKEN = {
    "gpt-3.5-turbo-0125": {
        "input": 5e-7,
        "output": 1.5e-6
    },
    "gpt-4-0125-preview": {
        "input": 1e-5,
        "output": 3e-6
    }
}


class OpenAIModel():
    
    def __init__(self,
        name: str = "gpt-4o-2024-05-13",
        temperature: float = 0.5,
        seed: int = None,
        max_tokens: int = 1024,
        frequency_penalty: float = 0.0,
        max_tries: int = 5,
        logger_name: str = None,
        do_compute_cost: bool = False,
        **kwargs
    ):
        self.name = name
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.frequency_penalty = frequency_penalty
        self.logger_name = logger_name
        self.date = kwargs.get('date', date.today().strftime("%Y-%m-%d"))
        self.logger = set_logger(logger_name, self.date) if logger_name else None
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.max_tries = max_tries
        self.costs = []
        self.do_compute_cost = do_compute_cost

    def compute_cost(self, input_messages: List[Dict[str, str]], output_message: str) -> float:
        def num_tokens_from_string(s: str) -> int:
            encoding = tiktoken.encoding_for_model(self.name)
            return len(encoding.encode(s))
        input_cost = sum([num_tokens_from_string(m["content"]) for m in input_messages]) * COSTS_PER_TOKEN[self.name]["input"]
        output_cost = num_tokens_from_string(output_message.content) * COSTS_PER_TOKEN[self.name]["output"]
        return input_cost + output_cost

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, str]:
        for i in range(self.max_tries):
            try:
                response = self.client.chat.completions.create(
                    model = self.name,
                    messages = messages,
                    temperature = self.temperature,
                    max_tokens = self.max_tokens,
                    frequency_penalty = self.frequency_penalty,
                    seed=self.seed,
                    **kwargs
                )
                break
            except Exception as e:
                if i == self.max_tries - 1:
                    return f"Error: {e}"
                print(f"Error, retrying ({i}/{self.max_tries})... {e}")
                if self.logger:
                    self.logger.info(f"Error, retrying ({i}/{self.max_tries})... {e}")
                time.sleep(10)
        
        if self.do_compute_cost:
            self.costs.append(self.compute_cost(messages, response.choices[0].message))

        result = response.choices[0].message.content
        
        if self.logger:
            self.logger.info("*"*200)
            cost = f"- COST: {self.costs[-1]:.5f}" if self.do_compute_cost else ""
            self.logger.info(f"GPT QUERY - ENGINE: {self.name} - TEMPERATURE: {self.temperature} - MAX TOKENS: {self.max_tokens} - FREQUENCY PENALTY: {self.frequency_penalty} {cost}")
            self.logger.info("\n".join([str(m) for m in messages]))
            self.logger.info("GPT RESPONSE")
            self.logger.info(result)
            self.logger.info("*"*200)
            
        return result

def add_thought(df: pd.DataFrame):
    model = OpenAIModel(logger_name="few_shot", do_compute_cost=False)
    pbar = tqdm(total=len(df[df["thought"] == ""]))
    def _add_thought_row(row: pd.Series):
        if len(row["thought"]) > 0:
            return row["thought"]
        question = row["user"]
        hint = HINT_STRING.format(answer=row["answer"])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(question=question, hint=hint)}
        ]
        for i in range(3):
            answer = model.generate(messages)
            try:
                thought = answer.split("Thought: ")[1].split("Answer: ")[0]
                break
            except:
                continue
        pbar.update(1)
        return thought
    df["thought"] = df.apply(_add_thought_row, axis=1)
    return df


if __name__ == "__main__":
    path = "/home/cd2853/rational_metareasoning/data/few_shot_prompts.json"
    df = pd.read_json(path)
    df = add_thought(df)
    save_to(df, path)
    