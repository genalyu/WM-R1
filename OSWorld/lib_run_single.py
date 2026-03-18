import datetime
import time
import json
import logging
import os

from wrapt_timeout_decorator import *

logger = logging.getLogger("desktopenv.experiment")


def run_single_example(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    runtime_logger = setup_logger(example, example_result_dir)
    agent.reset(runtime_logger)
    # t0 = time.time()
    obs = env.reset(task_config=example)
    # t1 = time.time()
    # print("reset time:", t1 - t0)
    # return 

    action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
    with open(os.path.join(example_result_dir, f"step_reset_{action_timestamp}.png"), "wb") as _f:
        _f.write(obs['screenshot'])
    
    with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
        traj_json = {
            "step_num": 0,
            'instruction': instruction,
            "action_timestamp": action_timestamp,
            "action": "reset",
            "reward": 0,
            "done": False,
            "info": {},
            "screenshot_file": f"step_reset_{action_timestamp}.png"
        }
        f.write(json.dumps(traj_json))
        f.write("\n")
    
    done = False
    step_idx = 0
    env.controller.start_recording()
    while not done and step_idx < max_steps:
        response, actions, logs = agent.predict(
            instruction,
            obs
        )
        for action in actions:
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Step %d: %s", step_idx + 1, action)
            obs, reward, done, info = env.step(action, args.sleep_after_execution)

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)
            # Save screenshot and trajectory information
            with open(os.path.join(example_result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"),
                      "wb") as _f:
                _f.write(obs['screenshot'])
            
            # Save trajectory information
            if isinstance(logs, dict):
                if "plan_result_full" in logs:
                    with open(os.path.join(example_result_dir, f"plan_result_full-step_{step_idx + 1}_{action_timestamp}.txt"),
                            "w") as _f:
                        _f.write(logs["plan_result_full"])
                if "plan_result" in logs:
                    with open(os.path.join(example_result_dir, f"plan_result-step_{step_idx + 1}_{action_timestamp}.txt"),
                            "w") as _f:
                        _f.write(logs["plan_result"])

            with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                traj_json = {
                    "step_num": step_idx + 1,
                    # 'instruction': instruction,
                    "prediction": response,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png"
                }
                if isinstance(logs, dict):
                    traj_json.update(logs)
                f.write(json.dumps(traj_json))
                f.write("\n")
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")
    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))


def setup_logger(example, example_result_dir):
    runtime_logger = logging.getLogger(f"desktopenv.example.{example['id']}")
    runtime_logger.setLevel(logging.DEBUG)
    runtime_logger.addHandler(logging.FileHandler(os.path.join(example_result_dir, "runtime.log")))
    return runtime_logger
