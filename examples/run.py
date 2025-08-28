from inferencesh import Inference, TaskStatus  # type: ignore


def run_with_updates() -> None:
    """Example showing how to get streaming updates."""
    api_key = "1nfsh-7yxm9j9mdpkkpsab2dxtnddxft"
    client = Inference(api_key=api_key, base_url="https://api-dev.inference.sh")

    try:
        # Run with stream=True to get updates
        for update in client.run(
            {
                "app": "lginf/llm-router",
                "input": {"image": "https://storage.googleapis.com/folip-api-images/images/rGF6LfQuGQUEox9YF3JkuOiITUm1/dc4c0e18cb7a4f669bc6b6f3b99e6147.png"},
                "infra": "cloud",
                "variant": "default"
            },
            stream=True  # Enable streaming updates
        ):
            # Print each update as it comes in
            status = update.get("status")
            status_name = TaskStatus(status).name if status is not None else "UNKNOWN"
            print(f"Status: {status_name}")
            
            # Print output when task completes
            if status == TaskStatus.COMPLETED:
                print("\n✓ Task completed!")
                print(f"Output: {update.get('output')}")
                break
            elif status == TaskStatus.FAILED:
                print(f"\n✗ Task failed: {update.get('error')}")
                break
            elif status == TaskStatus.CANCELLED:
                print("\n✗ Task was cancelled")
                break

    except Exception as exc:  # noqa: BLE001
        print(f"\nError: {type(exc).__name__}: {exc}")
        raise  # Re-raise to see full traceback


def run_simple() -> None:
    """Example showing simple synchronous usage."""
    api_key = "1nfsh-7yxm9j9mdpkkpsab2dxtnddxft"
    client = Inference(api_key=api_key, base_url="https://api-dev.inference.sh")

    try:
        # Simple synchronous run
        task = client.run(
            {
                "app": "lginf/llm-router",
                "input": {"image": "https://storage.googleapis.com/folip-api-images/images/rGF6LfQuGQUEox9YF3JkuOiITUm1/dc4c0e18cb7a4f669bc6b6f3b99e6147.png"},
                "infra": "cloud",
                "variant": "default"
            }
        )
        
        print(f"Task ID: {task.get('id')}")

        # Print final task
        if task.get("status") == TaskStatus.COMPLETED:
            print("\n✓ Task completed successfully!")
            print(f"Output: {task.get('output')}")
        else:
            status = task.get("status")
            status_name = TaskStatus(status).name if status is not None else "UNKNOWN"
            print(f"\n✗ Task did not complete. Final status: {status_name}")

    except Exception as exc:  # noqa: BLE001
        print(f"\nError: {type(exc).__name__}: {exc}")
        raise  # Re-raise to see full traceback


if __name__ == "__main__":
    # Choose which example to run:
    # run_with_updates()  # Shows streaming updates
    run_simple()      # Shows simple synchronous usage
