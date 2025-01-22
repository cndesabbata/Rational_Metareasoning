# Rational Metareasoning for Large Language Models

This repository contains the code for the paper ["Rational Metareasoning for Large Language Models"](http://arxiv.org/abs/2410.05563v1).

## Abstract
> Being prompted to engage in reasoning has emerged as a core technique for using large language models (LLMs), deploying additional inference-time compute to improve task performance. However, as LLMs increase in both size and adoption, inference costs are correspondingly becoming increasingly burdensome. How, then, might we optimize reasoning's cost-performance tradeoff? This work introduces a novel approach based on computational models of metareasoning used in cognitive science, training LLMs to selectively use intermediate reasoning steps only when necessary. We first develop a reward function that incorporates the Value of Computation by penalizing unnecessary reasoning, then use this reward function with Expert Iteration to train the LLM. Compared to few-shot chain-of-thought prompting and STaR, our method significantly reduces inference costs (20-37% fewer tokens generated across three models) while maintaining task performance across diverse datasets.

If you find this work useful for your research, please consider citing:
```
@misc{desabbata2024rationalmetareasoninglargelanguage,
      title={Rational Metareasoning for Large Language Models}, 
      author={C. Nicolò De Sabbata and Theodore R. Sumers and Thomas L. Griffiths},
      year={2024},
      eprint={2410.05563},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.05563}, 
}
```
