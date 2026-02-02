"""
MBPP (Mostly Basic Python Problems) pass@k evaluation.
427 sanitized problems, same methodology as HumanEval.
"""

import re
import math
from datasets import load_dataset
from nanocode.execution import execute_code
from tasks.common import Task


def extract_program(completion):
    """Extract Python code from LLM completion."""
    pattern = r'```(?:python)?\s*\n(.*?)\n```'
    matches = re.findall(pattern, completion, re.DOTALL)
    if matches:
        return matches[0].strip()
    return completion.strip()


def pass_at_k(n, c, k):
    """Unbiased estimator of pass@k."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


class MBPP(Task):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test").shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        prompt = row['prompt']
        code = row['code']
        test_list = row['test_list']

        # Build prompt: task description as user message, solution as assistant
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": code},
        ]
        conversation = {
            "messages": messages,
            "test_list": test_list,
        }
        return conversation

    def evaluate(self, conversation, completion):
        """Run the test assertions against the completion."""
        completion_code = extract_program(completion)
        test_code = "\n".join(conversation['test_list'])
        program = completion_code + "\n\n" + test_code
        result = execute_code(program, timeout=10.0)
        return result.success
