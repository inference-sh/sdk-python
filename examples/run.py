import asyncio
import os
from datetime import datetime
from typing import List

from inferencesh import Inference, TaskStatus

def main() -> None:
    api_key = "YOUR_INFERENCESH_API_KEY"
    client = Inference(api_key=api_key)
    
    app = "infsh/text-templating"

    try:
        result = client.run_sync(
            {
                "app": app,
                "input": {
                    "template": "{1} / {2}",
                    "strings": [
                        "god",
                        "particle",
                    ]
                },
                "worker_selection_mode": "private",
            },
        )
        

        # Print final result
        if result.get("status") == TaskStatus.COMPLETED:
            print(f"\n✓ task completed successfully!")
            print(f"result: {result.get('output', {}).get('result')}")
        else:
            status = result.get("status")
            status_name = TaskStatus(status).name if status is not None else "UNKNOWN"
            print(f"\n✗ task did not complete. final status: {status_name}")

    except Exception as exc:  # noqa: BLE001
        print(f"\nerror during run_sync: {type(exc).__name__}: {exc}")
        raise  # Re-raise to see full traceback

if __name__ == "__main__":
    main()
