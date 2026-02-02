"""
HumanEval + MBPP evaluation runner.
Supports distributed eval across GPUs.

Usage:
    python -m scripts.code_eval
    python -m scripts.code_eval --task=humaneval --num-samples=10 --temperature=0.8
    torchrun --nproc_per_node=8 -m scripts.code_eval
"""

import argparse
import math

import torch
import torch.distributed as dist

from nanocode.common import compute_init, compute_cleanup, get_dist_info, print0
from nanocode.checkpoint_manager import load_model
from nanocode.engine import Engine

from tasks.humaneval import HumanEval, pass_at_k as humaneval_pass_at_k
from tasks.mbpp import MBPP, pass_at_k as mbpp_pass_at_k


def run_pass_at_k_eval(task_object, tokenizer, engine, num_samples, max_new_tokens,
                        temperature, top_k, max_problems=None):
    """
    Generate num_samples completions per problem, compute pass@k.
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = next(iter(engine.model.parameters())).device

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    # Track per-problem results
    local_results = []  # list of (num_correct, num_total) tuples

    for i in range(ddp_rank, num_problems, ddp_world_size):
        conversation = task_object[i]
        encoded_prompt = tokenizer.render_for_completion(conversation)

        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        prefix_length = len(encoded_prompt)
        completions = [tokenizer.decode(result_tokens[prefix_length:]) for result_tokens in results]
        outcomes = [task_object.evaluate(conversation, completion) for completion in completions]
        num_correct = sum(outcomes)

        local_results.append((num_correct, num_samples))
        running_pass1 = sum(1 for c, n in local_results if c > 0) / len(local_results)
        print(f"\r\033[KRank {ddp_rank} | {len(local_results)}/{(num_problems + ddp_world_size - 1) // ddp_world_size} | pass@1 est: {100*running_pass1:.1f}%", end='', flush=True)

    print()

    # Aggregate across ranks
    if ddp:
        # Gather all results to rank 0
        all_results_list = [None] * ddp_world_size
        dist.all_gather_object(all_results_list, local_results)
        all_results = []
        for results in all_results_list:
            all_results.extend(results)
    else:
        all_results = local_results

    # Compute pass@k metrics
    if ddp_rank == 0 or not ddp:
        k_values = [1, 5, 10] if num_samples >= 10 else [1]
        metrics = {}
        for k in k_values:
            if k > num_samples:
                continue
            pass_k_values = []
            for num_correct, num_total in all_results:
                pass_k_values.append(humaneval_pass_at_k(num_total, num_correct, k))
            avg_pass_k = sum(pass_k_values) / len(pass_k_values)
            metrics[f"pass@{k}"] = avg_pass_k
            print0(f"  pass@{k}: {100*avg_pass_k:.2f}% ({len(all_results)} problems)")
        return metrics
    return {}


def main():
    parser = argparse.ArgumentParser(description='Code evaluation runner')
    parser.add_argument('-i', '--source', type=str, default="sft", help="Model source: base|sft")
    parser.add_argument('-a', '--task', type=str, default="all", help="Task: humaneval|mbpp|all")
    parser.add_argument('-n', '--num-samples', type=int, default=1, help="Samples per problem (default: 1)")
    parser.add_argument('-t', '--temperature', type=float, default=0.0, help="Sampling temperature")
    parser.add_argument('-k', '--top-k', type=int, default=50, help="Top-k sampling")
    parser.add_argument('-m', '--max-new-tokens', type=int, default=512, help="Max tokens to generate")
    parser.add_argument('-x', '--max-problems', type=int, default=None, help="Max problems to evaluate")
    parser.add_argument('-g', '--model-tag', type=str, default=None)
    parser.add_argument('-s', '--step', type=int, default=None)
    args = parser.parse_args()

    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    model, tokenizer, meta = load_model(args.source, device, phase="eval",
                                         model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)

    tasks_to_run = []
    if args.task in ["humaneval", "all"]:
        tasks_to_run.append(("HumanEval", HumanEval()))
    if args.task in ["mbpp", "all"]:
        tasks_to_run.append(("MBPP", MBPP()))

    all_metrics = {}
    for task_name, task_object in tasks_to_run:
        print0(f"\n{'='*50}")
        print0(f"Evaluating {task_name} ({len(task_object)} problems, {args.num_samples} samples each)")
        print0(f"{'='*50}")

        with autocast_ctx:
            metrics = run_pass_at_k_eval(
                task_object, tokenizer, engine,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature if args.num_samples > 1 else 0.0,
                top_k=args.top_k,
                max_problems=args.max_problems,
            )
        for k, v in metrics.items():
            all_metrics[f"{task_name}/{k}"] = v

    if ddp_rank == 0:
        print0(f"\n{'='*50}")
        print0("Summary:")
        for k, v in all_metrics.items():
            print0(f"  {k}: {100*v:.2f}%")

    compute_cleanup()


if __name__ == "__main__":
    main()
