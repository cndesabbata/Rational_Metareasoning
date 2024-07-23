import torch
import logging

from policy_model import AgentPretrainedModel
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import logprobs_from_logits

class RewardModel():

    def __init__(self,
            model: AgentPretrainedModel,
            voc_gamma: float = 1e-2,
            logger: logging.Logger = None,
            device: torch.device = None,
            sample_size: int = 4,
            tokenizer: AutoTokenizer = None,
        ):
        self.model = model
        self.voc_gamma = voc_gamma
        self.logger = logger
        self.tokenizer = tokenizer
        self.sample_size = sample_size+1
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_scores(
            self, 
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            target_mask: torch.Tensor,
            thoughts_mask: torch.Tensor,
            **kwargs
        ) -> torch.Tensor:
        ### Compute target probabilities
        target_probabilities = RewardModel.compute_target_probs(
            model=self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_mask=target_mask,
        )
        torch.set_printoptions(precision=4, sci_mode=False)
        self.logger.info(f"Scores: {target_probabilities.view(-1, self.sample_size)}")
        ### Compute utility score (probability increment)
        scores = (target_probabilities.view(-1, self.sample_size) - target_probabilities.view(-1, self.sample_size)[:, -1].unsqueeze(1)).view(-1)
        self.logger.info(f"Scores: {scores.view(-1, self.sample_size)}")
        ### Compute penalties 
        lengths = thoughts_mask.sum(dim=1).float().view(-1, self.sample_size)
        penalties = self.voc_gamma * ((lengths - lengths.min(dim=1).values.unsqueeze(1))).view(-1).to(self.device)
        ### Compute rewards
        rewards = scores - penalties
        ### Log results
        self.logger.info(f"Lenghts: Mean {lengths.mean():.4f}, Std {lengths.std():.4f}, Max {lengths.max():.4f}, Min {lengths.min():.4f}")
        self.logger.info(f"Penalties: Mean {penalties.mean():.4f}, Std {penalties.std():.4f}, Max {penalties.max():.4f}, Min {penalties.min():.4f}")
        self.logger.info(f"Scores: Mean {scores.mean():.4f}, Std {scores.std():.4f}, Max {scores.max():.4f}, Min {scores.min():.4f}")
        decoded_inputs = [self.tokenizer.decode(i, skip_special_tokens=False) for i in input_ids]
        zipped_results = list(zip(decoded_inputs, scores.view(-1), penalties.view(-1), rewards.view(-1)))
        for i, (decoded_input, score, penalty, score_penalized) in enumerate(zipped_results):
            self.logger.info(f"Input: {decoded_input}\nScore: {score:.4f}, Penalty: {penalty:.4f}, Score penalized: {score_penalized:.4f}\n")
        ### Return rewards
        return rewards
    
    __call__ = compute_scores
    
    @staticmethod
    def compute_target_probs(
        model: AutoModelForCausalLM,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ):
        ''' Compute the probability of the completion given the prompt.'''
        ### Compute logits
        with torch.no_grad():
            outputs = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            logits = outputs.logits[:, :-1, :].contiguous()
        ### Gather logprobs
        target = torch.where(target_mask==0, torch.ones_like(input_ids).to(model.device)*model.tokenizer.pad_token_id, input_ids)[:, 1:]
        target_mask = target_mask[:, 1:]
        logprobs = logprobs_from_logits(logits, target)
        probabilities = torch.tensor([torch.exp(lp[m.bool()][2:-3]).prod() for lp, m in zip(logprobs, target_mask)]).to(model.device)
        return probabilities