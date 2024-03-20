import argparse
import numpy as np
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError

from eval.grader import *
from utils.parser import *
from utils.utils import load_jsonl
from utils.python_executor import PythonExecutor


def evaluate(data_name, prompt_type, samples: list=None, file_path: str=None, max_num_samples=None, use_train_prompt_format=False, code_concat = False, max_func_call = 4, code_exec_warning = False, max_code_fix_retries = 4, execute=False):
    assert samples or file_path, "samples or file_path must be provided"
    if not samples:
        samples = list(load_jsonl(file_path))
    # dedup by idx
    if 'idx' in samples[0]:
        samples = {sample['idx']: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x: x['idx']) 
    else:
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]

    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]
    
    # parse gt
    for sample in samples:
        sample['gt_cot'], sample['gt'] = parse_ground_truth(sample, data_name)

    # execute
    if ('pred' not in samples[0]) or execute:
        if "pal" in prompt_type:
            executor = PythonExecutor(get_answer_expr="solution()")
        else:
            executor = PythonExecutor(get_answer_from_stdout=True)

        for sample in tqdm(samples, desc="Execute"):
            sample['pred'] = []
            sample['report'] = []
            for code in sample['code']:
                pred, report = run_execute(executor, code, prompt_type, execute=True)
                sample['pred'].append(pred)
                sample['report'].append(report)

    params = [(idx, pred, sample['gt']) for idx, sample in enumerate(samples) for pred in sample['pred']]

    # Print the list of empty samples with no predictions
    empty_samples = [s for s in samples if not s['pred'][-1]]
    if empty_samples:
        print(f"Empty samples: {len(empty_samples)}")
        for sample in empty_samples:
            print("Idx: ", sample['idx'])
            # print(sample)

    scores = []
    timeout_cnt = 0 

    with ProcessPool() as pool:
        future = pool.map(math_equal_process, params, timeout=3)
        iterator = future.result()
        with tqdm(total=len(samples), desc="Evaluate") as progress_bar:
            while True:
                try:
                    result = next(iterator)
                    scores.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    scores.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    print(error.traceback)
                    exit()
                progress_bar.update(1) 

    idx = 0
    score_mat = []
    for sample in samples:
        sample['score'] = scores[idx: idx+len(sample['pred'])]
        assert len(sample['score']) == len(sample['pred'])
        score_mat.append(sample['score'])
        idx += len(sample['pred'])

    max_len = max([len(s) for s in score_mat])

    for i, s in enumerate(score_mat):
        if len(s) < max_len:
            score_mat[i] = s + [s[-1]] * (max_len - len(s)) # pad

    # output mean of each column of scores
    col_means= np.array(score_mat).mean(axis=0)
    mean_score = list(np.round(col_means * 100, decimals=1))

    result_str = f"Num samples: {len(samples)}\n" \
        f"Num scores: {len(scores)}\n" \
        f"Timeout samples: {timeout_cnt}\n" \
        f"Empty samples: {len([s for s in samples if not s['pred'][-1]])}\n" \
        f"Prompt type: {prompt_type}\n" \
        f"use_train_prompt_format: {use_train_prompt_format}\n" \
        f"code_concat: {code_concat}\n" \
        f"max_func_call: {max_func_call}\n" \
        f"code_exec_warning: {code_exec_warning}\n"\
        f"max_code_fix_retries: {max_code_fix_retries}\n"\
        f"Mean score: {mean_score}\n"

    # each type score
    if "type" in samples[0]:
        type_scores = {}
        for sample in samples:
            if sample['type'] not in type_scores:
                type_scores[sample['type']] = []
            type_scores[sample['type']].append(sample['score'][-1])
        type_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in type_scores.items()}
        type_scores = {k: v for k, v in sorted(type_scores.items(), key=lambda item: item[0])}
        result_str += f"Type scores: {type_scores}\n"

    # each subject score
    if "subject" in samples[0]:
        subject_scores = {}
        for sample in samples:
            if sample['subject'] not in subject_scores:
                subject_scores[sample['subject']] = []
            subject_scores[sample['subject']].append(sample['score'][-1])
        subject_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in subject_scores.items()}
        subject_scores = {k: v for k, v in sorted(subject_scores.items(), key=lambda item: item[0])}
        result_str += f"Type scores: {subject_scores}\n"

    # each level score
    if "level" in samples[0]:
        level_scores = {}
        for sample in samples:
            if sample['level'] not in level_scores:
                level_scores[sample['level']] = []
            level_scores[sample['level']].append(sample['score'][-1])
        level_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in level_scores.items()}
        level_scores = {k: v for k, v in sorted(level_scores.items(), key=lambda item: item[0])}
        result_str += f"Level scores: {level_scores}\n"

    print(result_str)
    return result_str


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="math")
    parser.add_argument("--prompt_type", type=str, default="tora")
    parser.add_argument("--file_path", type=str, default=None, required=True)
    parser.add_argument("--max_num_samples", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    evaluate(data_name=args.data_name, prompt_type=args.prompt_type, file_path=args.file_path,
             max_num_samples=args.max_num_samples, execute=args.execute)
