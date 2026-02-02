"""
Base class for all Tasks.
A Task is a dataset of problems with evaluation criteria.
"""

import random


class Task:
    """Base class of a Task with lightweight slicing."""

    def __init__(self, start=0, stop=None, step=1):
        assert start >= 0
        assert stop is None or stop >= start
        assert step >= 1
        self.start = start
        self.stop = stop
        self.step = step

    @property
    def eval_type(self):
        raise NotImplementedError

    def num_examples(self):
        raise NotImplementedError

    def get_example(self, index):
        raise NotImplementedError

    def __len__(self):
        start = self.start
        stop = self.num_examples() if self.stop is None else self.stop
        step = self.step
        span = stop - start
        num = (span + step - 1) // step
        assert num >= 0
        return num

    def __getitem__(self, index: int):
        assert isinstance(index, int)
        physical_index = self.start + index * self.step
        return self.get_example(physical_index)

    def evaluate(self, problem, completion):
        raise NotImplementedError


class TaskMixture(Task):
    """
    Mix multiple task datasets for SFT training.
    Deterministically shuffled so tasks are interleaved throughout training.
    """

    def __init__(self, tasks, **kwargs):
        super().__init__(**kwargs)
        self.tasks = tasks
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum(self.lengths)
        self.index_map = []
        for task_idx, task_length in enumerate(self.lengths):
            for local_idx in range(task_length):
                self.index_map.append((task_idx, local_idx))
        rng = random.Random(42)
        rng.shuffle(self.index_map)

    def num_examples(self):
        return self.num_conversations

    def get_example(self, index):
        assert 0 <= index < self.num_conversations
        task_idx, local_idx = self.index_map[index]
        return self.tasks[task_idx][local_idx]
