"""Real API test for sync and async clients."""
import asyncio
from inferencesh import Inference, AsyncInference, TaskStatus

API_KEY = "1nfsh-47p5ztz7qzm6j32maygz51eqbx"

TASK_PARAMS = {
    "app": "infsh/text-templating",
    "input": {
        "template": "hello {1}",
        "strings": ["world"]
    },
    "infra": "cloud",
    "variant": "default"
}


def test_sync():
    """Test synchronous client."""
    print("=" * 50)
    print("SYNC CLIENT TEST")
    print("=" * 50)
    
    client = Inference(api_key=API_KEY, base_url="https://api.inference.sh")
    
    # Test 1: Run and wait (default)
    print("\n1. run() - wait for completion (default)")
    task = client.run(TASK_PARAMS)
    print(f"   Task ID: {task['id']}")
    print(f"   Status: {TaskStatus(task['status']).name}")
    if task["status"] == TaskStatus.COMPLETED:
        print(f"   Output: {task['output']}")
    
    # Test 2: Run with wait=False
    print("\n2. run(wait=False) - return immediately")
    task = client.run(TASK_PARAMS, wait=False)
    print(f"   Task ID: {task['id']}")
    print(f"   Status: {TaskStatus(task['status']).name}")
    
    # Test 3: get_task
    print(f"\n3. get_task('{task['id']}')")
    task_info = client.get_task(task["id"])
    print(f"   Status: {TaskStatus(task_info['status']).name}")
    
    # Test 4: Stream updates
    print("\n4. run(stream=True) - stream updates")
    for update in client.run(TASK_PARAMS, stream=True):
        status = update.get('status')
        if status is not None:
            status_name = TaskStatus(status).name
            print(f"   Status: {status_name}")
            if status == TaskStatus.COMPLETED:
                print(f"   Output: {update.get('output')}")
                break
    
    # Test 5: stream_task
    print("\n5. stream_task() - stream existing task")
    task = client.run(TASK_PARAMS, wait=False)
    with client.stream_task(task["id"]) as stream:
        for update in stream:
            status = update.get('status')
            if status is not None:
                status_name = TaskStatus(status).name
                print(f"   Status: {status_name}")
                if status == TaskStatus.COMPLETED:
                    print(f"   Output: {update.get('output')}")
                    break
    
    print("\n✓ Sync client tests passed!")


async def test_async():
    """Test asynchronous client."""
    print("\n" + "=" * 50)
    print("ASYNC CLIENT TEST")
    print("=" * 50)
    
    client = AsyncInference(api_key=API_KEY, base_url="https://api.inference.sh")
    
    # Test 1: Run and wait (default)
    print("\n1. await run() - wait for completion (default)")
    task = await client.run(TASK_PARAMS)    
    print(f"   Task ID: {task['id']}")
    print(f"   Status: {TaskStatus(task['status']).name}")
    if task["status"] == TaskStatus.COMPLETED:
        print(f"   Output: {task['output']}")
    
    # Test 2: Run with wait=False
    print("\n2. await run(wait=False) - return immediately")
    task = await client.run(TASK_PARAMS, wait=False)
    print(f"   Task ID: {task['id']}")
    print(f"   Status: {TaskStatus(task['status']).name}")
    
    # Test 3: get_task
    print(f"\n3. await get_task('{task['id']}')")
    task_info = await client.get_task(task["id"])
    print(f"   Status: {TaskStatus(task_info['status']).name}")
    
    # Test 4: Stream updates
    print("\n4. async for in await run(stream=True)")
    async for update in await client.run(TASK_PARAMS, stream=True):
        status = update.get('status')
        if status is not None:
            status_name = TaskStatus(status).name
            print(f"   Status: {status_name}")
            if status == TaskStatus.COMPLETED:
                print(f"   Output: {update.get('output')}")
                break
    
    # Test 5: stream_task
    print("\n5. async with stream_task()")
    task = await client.run(TASK_PARAMS, wait=False)
    async with client.stream_task(task["id"]) as stream:
        async for update in stream:
            status = update.get('status')
            if status is not None:
                status_name = TaskStatus(status).name
                print(f"   Status: {status_name}")
                if status == TaskStatus.COMPLETED:
                    print(f"   Output: {update.get('output')}")
                    break
    
    print("\n✓ Async client tests passed!")


if __name__ == "__main__":
    # Run sync tests
    test_sync()
    
    # Run async tests
    asyncio.run(test_async())
    
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)
