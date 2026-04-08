import os
import sys

# Ensure project root is in path
ROOT = os.getcwd()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from openenv_logic.environment import OpenEnvEnvironment, TaskID, Action

def test_task(task_id):
    print(f"\n--- Testing Task: {task_id} ---")
    env = OpenEnvEnvironment(task_id=task_id)
    obs = env.reset()
    print(f"Initial state: {obs.state_description}")
    
    steps = 0
    while not obs.done and steps < 100:
        # Simple heuristic: pick the first available action
        # For CodeReview, it cycles through stages automatically.
        # For Scheduler, it picks the first schedule or cancel.
        action_name = obs.available_actions[0]
        action = Action(action_type=action_name, parameters={})
        
        step_result = env.step(action)
        obs = step_result.observation
        steps += 1
        
        # print(f"Step {steps}: {action_name} -> {obs.state_description}")
    
    print(f"Task {task_id} completed in {steps} steps.")
    print(f"Final Score: {env.final_score()}")
    print(f"Cumulative Reward: {obs.context.get('cumulative_reward', 'N/A')}")
    print("Success: No crashes.")

if __name__ == "__main__":
    try:
        test_task(TaskID.EMAIL_TRIAGE)
        test_task(TaskID.CODE_REVIEW)
        test_task(TaskID.MEETING_SCHEDULER)
        print("\n✅ All 3 tasks verified locally without crashes.")
    except Exception as e:
        print(f"\n❌ CRASH DETECTED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
