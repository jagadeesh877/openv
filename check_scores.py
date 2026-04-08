from main import run_baseline_agent
import json

results = run_baseline_agent()
output = []
for task_id, data in results.items():
    output.append({
        "task": task_id,
        "score": data["final_score"],
        "reward": data["total_reward"],
        "steps": data["steps"]
    })

print(json.dumps(output, indent=2))
