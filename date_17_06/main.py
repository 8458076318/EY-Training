import os
import asyncio
from fastapi import FastAPI
from azure.servicebus import ServiceBusClient, ServiceBusMessage
from azure.cosmos import CosmosClient
from groq import Groq

app = FastAPI(title="Pipeline API with Groq")

QUEUE_NAME = "ragqueue"


def get_env(name: str) -> str | None:
    return os.getenv(name)


def get_container():
    cosmos_uri = get_env("COSMOS_URI")
    cosmos_key = get_env("COSMOS_KEY")
    if not cosmos_uri or not cosmos_key:
        return None

    cosmos_client = CosmosClient(cosmos_uri, credential=cosmos_key)
    db = cosmos_client.get_database_client("ragprojectdb")
    return db.get_container_client("Items")


def get_groq_client():
    groq_api_key = get_env("GROQ_API_KEY")
    if not groq_api_key:
        return None

    return Groq(api_key=groq_api_key)

@app.post("/ingest")
async def ingest_data(payload: dict):
    """Step 1: Input Queue - Accepts data and pushes it onto the queue."""
    sb_conn_str = get_env("SERVICE_BUS_CONNECTION_STRING")
    if not sb_conn_str:
        return {"error": "SERVICE_BUS_CONNECTION_STRING is not set"}

    async with ServiceBusClient.from_connection_string(sb_conn_str) as client:
        sender = client.get_queue_sender(queue_name=QUEUE_NAME)
        message = ServiceBusMessage(str(payload))
        await sender.send_messages(message)
    return {"status": "Message queued successfully"}

@app.get("/extend/{item_id}")
async def extend_with_groq(item_id: str):
    """Step 4 & 5: Fetch from Store, FastAPI processing, and Extend via Groq API."""
    container = get_container()
    groq_client = get_groq_client()

    if container is None:
        return {"error": "COSMOS_URI and COSMOS_KEY must be set"}

    if groq_client is None:
        return {"error": "GROQ_API_KEY must be set"}

    # Read raw document from Cosmos DB
    item = container.read_item(item=item_id, partition_key=item_id)
    
    # Send processing request to Groq using an open-weights model
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user", 
                "content": f"Summarize and analyze this data cleanly: {item}"
            }
        ],
        temperature=0.2,
        max_tokens=1024
    )
    
    return {
        "original_data": item,
        "groq_analysis": response.choices[0].message.content
    }

async def queue_processor():
    """Step 2 & 3: Background processor reading from Queue and writing to Store."""
    sb_conn_str = get_env("SERVICE_BUS_CONNECTION_STRING")
    if not sb_conn_str:
        print("Processor skipped: SERVICE_BUS_CONNECTION_STRING is not set")
        return

    container = get_container()
    if container is None:
        print("Processor skipped: COSMOS_URI and COSMOS_KEY must be set")
        return

    while True:
        try:
            async with ServiceBusClient.from_connection_string(sb_conn_str) as client:
                receiver = client.get_queue_receiver(queue_name=QUEUE_NAME)
                async with receiver:
                    messages = await receiver.receive_messages(max_message_count=1, max_wait_time=5)
                    for msg in messages:
                        data_str = str(msg)
                        # Process and write to Cosmos DB
                        container.upsert_item({"id": str(msg.sequence_number), "content": data_str})
                        await receiver.complete_message(msg)
        except Exception as e:
            print(f"Processor Error: {e}")
        await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event():
    # Start the queue processor loop asynchronously in the background
    asyncio.create_task(queue_processor())
