import asyncio
import os
from datetime import datetime
from typing import List

from inferencesh import Inference, TaskStatus


def main() -> None:
    api_key = "1nfsh-0fd8bxd5faawt9m0cztdym4q6s"
    client = Inference(api_key=api_key, base_url="https://api-dev.inference.sh")

    app = "infsh/text-templating"

    try:
        task = client.run(
            {
                "app": "infsh/lightning-wan-2-2-i2v-a14b",
                "input": {
                    "negative_prompt": "oversaturated, overexposed, static, blurry details, subtitles, stylized, artwork, painting, still image, overall gray, worst quality, low quality, JPEG artifacts, ugly, deformed, extra fingers, poorly drawn hands, poorly drawn face, malformed, disfigured, deformed limbs, fused fingers, static motionless frame, cluttered background, three legs, crowded background, walking backwards",
                    "prompt": "test",
                    "num_frames": 81,
                    "num_inference_steps": 4,
                    "fps": 16,
                    "boundary_ratio": 0.875,
                    "image": "https://images.dev.letz.ai/5ed74083-f9d1-4897-b8e3-c8f1596af767/fa6b9cbc-9465-4fe8-b5ba-08c7a75d4975/drawing_extreme_closeup_portrait_of_junck37342762320240205225633.jpg",
                },
                "infra": "private",
                # "workers": [],
                "variant": "fp16_480p",
            }
        )
        
        print(task["id"])

        # Print final task
        if task.get("status") == TaskStatus.COMPLETED:
            print(f"\n✓ task completed successfully!")
            print(f"task: {task.get('output', {}).get('task')}")
        else:
            status = task.get("status")
            status_name = TaskStatus(status).name if status is not None else "UNKNOWN"
            print(f"\n✗ task did not complete. final status: {status_name}")

    except Exception as exc:  # noqa: BLE001
        print(f"\nerror during run_sync: {type(exc).__name__}: {exc}")
        raise  # Re-raise to see full traceback


if __name__ == "__main__":
    main()
