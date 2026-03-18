
import os
import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='results_tasks_cot_0215_gpt4o_uitars_v2/pyautogui/screenshot/gpt-4o/')
parser.add_argument('--meta_path', type=str, default='./evaluation_examples/test_all.json')
parser.add_argument('--trials', type=int, nargs='+', default=[])
parser.add_argument('--detail', action='store_true')
parser.add_argument('--filter_fail', action='store_true')

args = parser.parse_args()

# with open('evaluation_examples/test_success_small.json', 'r') as f:
# with open('evaluation_examples/test_success.json', 'r') as f:
# with open('evaluation_examples/test_all.json', 'r') as f:
# with open('evaluation_examples/test_subset16_0.json', 'r') as f:
# with open('evaluation_examples/test_small.json', 'r') as f:
with open(args.meta_path, 'r') as f:
    data = json.load(f)

# for domain, datalist in data.items():
#     for d in datalist:
#         with open(f'./evaluation_examples/examples/{domain}/{d}.json', 'r') as f:
#             task_config = json.load(f)
#             if not task_config['id'] == d:
#                 print(f"Task {d}: {task_config['id'] == d}, task_config['id']: {task_config['id']}, d: {d}")
# import random
# data = {k: random.sample(v, int(len(v)*0.1)) for k, v in data.items()}

filter_fail = args.filter_fail
root_path = args.root_path

def load_result(subdir):
    if os.path.exists(os.path.join(subdir, 'traj.jsonl')) and os.path.exists(os.path.join(subdir, 'result.txt')):
        with open(os.path.join(subdir, 'result.txt'), 'r') as f:
            result = f.read().strip()
            if result.lower() == 'false':
                result = 0
            elif result.lower() == 'true':
                result = 1
            else:
                result = float(result)
            
        if filter_fail:
            with open(os.path.join(subdir, 'traj.jsonl'), 'r') as f:
                traj = f.readlines()
                traj = [json.loads(x) for x in traj]
                if traj[-1]['action'] == 'FAIL' and len(traj) >= 16:
                    result = 0
        return result
    return None

trials = args.trials

if len(trials) == 0:
    trials = [int(x) for x in os.listdir(root_path)]
    trials = sorted(trials)
print('trials:', trials)

total_num = sum([len(datalist) for datalist in data.values()])
acc = {}
for domain, datalist in data.items():
    for d in datalist:
        for i in trials:
            subdir1 = os.path.join(root_path, str(i), domain, d)

            r1 = load_result(subdir1)
            if r1 is None:
                continue

            if f'{domain}_{d}' not in acc:         
                acc[f'{domain}_{d}'] = [{'trial': i,'result': r1}]
            else:
                acc[f'{domain}_{d}'].append({'trial': i,'result': r1})


acc_list = [any([t['result'] > 0.5 for t in x]) for x in acc.values()]
final_acc = sum(acc_list) / len(acc_list)

print(f'Acc: {final_acc}, success: {sum(acc_list)}, valid task: {len(acc_list)}/{total_num}') 

print(f'Acc over all: {sum(acc_list) / total_num}')
if False:
# if True:
    task_config_subset = {}
    for domain, datalist in data.items():
        for task_id in datalist:
            if f'{domain}_{task_id}' not in acc:
                print(f'{domain}, {task_id}')
                continue
            cur_acc = any([t['result'] > 0.5 for t in acc[f'{domain}_{task_id}']])
            num_success = sum([t['result'] > 0.5 for t in acc[f'{domain}_{task_id}']])
            num_trials = len(acc[f'{domain}_{task_id}'])
            if num_success > 0:
                task_config_subset[domain] = task_config_subset.get(domain, [])
                task_config_subset[domain].append(task_id)

    # with open('./test_success_uitars1.5_wo_impossible.json', 'w') as f:
        # json.dump(task_config_subset, f, indent=4)

    exit()

if args.detail:
    for domain, datalist in data.items():
        for task_id in datalist:
            if f'{domain}_{task_id}' not in acc:
                continue
            cur_acc = any([t['result'] > 0.5 for t in acc[f'{domain}_{task_id}']])
            num_success = sum([t['result'] > 0.5 for t in acc[f'{domain}_{task_id}']])
            num_trials = len(acc[f'{domain}_{task_id}'])

            if cur_acc:
                with open(f'./evaluation_examples/examples/{domain}/{task_id}.json', 'r') as f:
                    task_config = json.load(f)
                    instruction = task_config['instruction']

                lengthes = []
                for trial_id in trials:
                    sub_dir = os.path.join(root_path, str(trial_id), domain, task_id)
                    if not os.path.exists(os.path.join(sub_dir, 'traj.jsonl')):
                        continue

                    with open(os.path.join(sub_dir, 'traj.jsonl'), 'r') as f:
                        traj = f.readlines()
                        traj_len = len(traj)
                    
                    trial_acc = load_result(sub_dir)
                    if trial_acc is None:
                        continue
                    if trial_acc > 0.5:
                        lengthes.append(str(traj_len) + '(✓)')
                    else:
                        lengthes.append(str(traj_len))

                print(f'{domain}_{task_id} is successful {num_success}/{num_trials} |traj length: [{", ".join(lengthes)}]: {instruction}')
        
    
def show_acc_per_domain(data, acc):
    for domain, datalist in data.items():
        num_tasks = len(datalist)
        total_valid_tasks = 0
        acc_per_domain = []
        for task_id in datalist:
            if f'{domain}_{task_id}' not in acc:
                acc_per_domain.append(0)
            else:
                total_valid_tasks += 1
                acc_per_domain.append(any([t['result'] > 0.5 for t in acc[f'{domain}_{task_id}']]))
        # print(f'{domain} acc: {sum(acc_per_domain) / num_tasks}, num tasks: {num_tasks}')
        print(f'{domain} acc: {sum(acc_per_domain) / max(total_valid_tasks, 1e-6)}, num tasks: {total_valid_tasks}/{num_tasks}')

    overall_acc = []
    for acc_ in acc.values():
        overall_acc.extend(acc_)

    print(f'Overall acc: {sum([t["result"] > 0.5 for t in overall_acc]) / len(overall_acc)}')
            

def show_acc_per_trial(trials, data, acc):
    num_valid = 0
    for domain, datalist in data.items():
        num_valid += len(datalist)

    mean_acc = []
    for trial_id in trials:
        acc_per_trial = []
        for domain, datalist in data.items():
            for task_id in datalist:
                if f"{domain}_{task_id}" not in acc:
                    continue
                result_list = acc[f"{domain}_{task_id}"]
                for result in result_list:
                    if result['trial'] == trial_id:
                        acc_per_trial.append(result['result'])
                
        # num_valid = len(acc_per_trial)
        acc_per_trial_ = sum([t > 0.5 for t in acc_per_trial]) / num_valid
    
        mean_acc.append(acc_per_trial_)  
        print(f'Trial {trial_id} acc: {acc_per_trial_}, num valid: {len(acc_per_trial)} / {num_valid}')
    
    print(f'Mean acc: {sum(mean_acc) / len(mean_acc)}')
                

show_acc_per_domain(data, acc)
show_acc_per_trial(trials, data, acc)

