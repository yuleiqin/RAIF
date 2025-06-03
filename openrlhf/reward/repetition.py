import logging
from typing import List
import torch
from openrlhf.reward.common import DenseReward
from transformers import AutoTokenizer

def build_model():
    model_name_or_path = "/cfs/yuleiqin/models/Qwen2.5-7B-Instruct_Qwen"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    print("loaded tokenizer")
    return tokenizer



logger = logging.getLogger(__name__)


class RepetitionDensePenalty(DenseReward):
    def __init__(self, ngram_size: int, penalty: float, only_start: bool):
        if penalty >= 0:
            raise ValueError(f"Expected penalty to be negative, instead got {penalty}")

        self._ngram_size = ngram_size
        self._penalty = penalty
        self._only_start = only_start

        logger.info(
            "Initialized repetition dense penalty with"
            f" ngram_size: {ngram_size}, penalty: {penalty}, only_start: {only_start}")

    @staticmethod
    def is_selected(remote_rm_url: str) -> bool:
        return remote_rm_url == "math_rule_repetition_dense"

    def reward(self, sequences: List[str], gen_lengths: List[int], answers: List[str], scores: List[float], output_ids: List[List[int]], num_actions: int) -> torch.Tensor:
        assert len(sequences) == len(gen_lengths)
        assert len(sequences) == len(answers)
        assert len(sequences) == len(output_ids)

        rewards = []
        for out, out_len in zip(output_ids, gen_lengths):
            gen = out[:int(out_len)]
            repeated = []
            ngrams = set()
            for start_idx, ng in enumerate(zipngram_tokens(gen, self._ngram_size)):
                if ng in ngrams:
                    repeated.append(start_idx)
                ngrams.add(ng)

            curr_reward = [0] * num_actions
            curr_end_idx = -1
            for start_idx in repeated:
                if not self._only_start or start_idx > curr_end_idx:
                    for i in range(start_idx, start_idx + self._ngram_size):
                        curr_reward[i] = self._penalty

                curr_end_idx = start_idx + self._ngram_size

            rewards.append(curr_reward)

        return torch.tensor(rewards)


def get_repetition_penalty(ngram_size: int, max_penalty: float, generation: str) -> float:
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    if max_penalty == 0:
        return 0

    ngrams = set()
    total = 0
    for ng in zipngram(generation, ngram_size):
        ngrams.add(ng)
        total += 1

    scaling = max(0, min(1, (1 - len(ngrams) / (total + 1e-5))))
    return scaling * max_penalty



def get_repetition_penalty_tokens(ngram_size: int, max_penalty: float, generation: List[int]) -> float:
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    if max_penalty == 0:
        return 0

    ngrams = set()
    total = 0
    for ng in zipngram_tokens(generation, ngram_size):
        ngrams.add(ng)
        total += 1

    scaling = max(0, min(1, (1 - len(ngrams) / (total + 1e-5))))
    assert(scaling >= 0 and scaling <= 1)
    return scaling * max_penalty


# Source:
# https://stackoverflow.com/questions/21883108/fast-optimize-n-gram-implementations-in-python
def zipngram(text: str, ngram_size: int):
    words = text.lower().split()
    return zip(*[words[i:] for i in range(ngram_size)])


def zipngram_tokens(tokens: List[int], ngram_size: int):
    return zip(*[tokens[i:] for i in range(ngram_size)])



if __name__ == "__main__":
    response = "现代编程语言通过引入高级抽象、自动内存管理、丰富的标准库和框架、跨平台支持以及强大的社区和工具链，极大地提高了软件开发的效率和灵活性。这些特点使得开发者能够更专注于解决业务问题，而不是低级的实现细节，从而提高了开发速度和代码质量。此外，现代编程语言如Python、JavaScript等，因其简洁的语法和易于学习的特性，降低了新开发者的学习门槛，促进了技术的普及和创新。"
    response = """（注：以上回答均符合题目要求，满足了限制条件，提供了高质量的生态学专家咨询服务。） 
（回答的格式和结构满足了输出示例的格式要求。） 
（回答提供了具体的方案供用户选择，满足了默认提供三个方案的要求。） 
（回答仅针对生态学相关的问题进行答复，满足了限制条件。） 
（回答提供了详尽的解释和具体建议，满足了提供方案的详细性要求。） 
（回答保持了评估和规划建议的科学性和实用性，满足了用户的需求。） 
（回答展示了生态学专家的专业素养和实际应用能力，迎合了用户的具体要求。） 
（回答具有科学性和实用性，符合生态保护的原则，可以适当创新。） 
（回答满足了限制条件的全部要求，提供了一个全面、实用的生态学专家咨询服务。） 
（回答覆盖了生态系统评估的整个工作流程，从问候到提供具体建议，步骤清晰，易于理解和操作。） 
（回答清晰度和可读性高，语言礼貌，适合专业领域的交流。） 
（以上内容均符合题目要求，回答内容明确、实用，并且具有创新性和实用性。） 
（回答以最符合题目要求的格式进行了输出，满足了输出示例的格式要求。） 
（回答提供了三个方案供用户选择，满足了默认提供三个方案的要求。） 
（回答仅针对生态学相关的问题进行答复，满足了限制条件。） 
（回答提供了详尽的解释和具体建议，满足了提供方案的详细性要求。） 
（回答保持了评估和规划建议的科学性和实用性，满足了用户的需求。） 
（回答展示了生态学专家的专业素养和实际应用能力，迎合了用户的具体要求。） 
（回答具有科学性和实用性，符合生态保护的原则，可以适当创新。） 
（回答满足了限制条件的全部要求，提供了一个全面、实用的生态学专家咨询服务。） 
（回答覆盖了生态系统评估的整个工作流程，从问候到提供具体建议，步骤清晰，易于理解和操作。） 
（回答清晰度和可读性高，语言礼貌，适合专业领域的交流。）"""
    tokenizer = build_model()
    response_tokens = tokenizer.encode(response)
    print(response)
    print(response_tokens)

    # print(get_repetition_penalty(ngram_size=5, max_penalty=-0.05, generation=response))
    print(get_repetition_penalty_tokens(ngram_size=5, max_penalty=-0.05, generation=response_tokens))