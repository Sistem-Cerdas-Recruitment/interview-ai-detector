import json
import os
from kafka import KafkaConsumer
from get_gpt_answer import GetGPTAnswer
from typing import List
from concurrent.futures import ThreadPoolExecutor


def get_gpt_responses(data: dict[str, any], gpt_helper: GetGPTAnswer):
    # data["gpt35_answer"] = gpt_helper.generate_gpt35_answer(data["question"])
    # data["gpt4_answer"] = gpt_helper.generate_gpt4_answer(data["question"])
    data["gpt35_answer"] = "This is gpt35 answer"
    data["gpt4_answer"] = "This is gpt4 answer"
    return data


def process_batch(batch: List[dict[str, any]], batch_size: int):
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        gpt_helper = GetGPTAnswer()
        futures = [executor.submit(
            get_gpt_responses, data, gpt_helper) for data in batch]
        results = [future.result() for future in futures]

    print("Batch ready with gpt responses", results)


def consume_messages():
    consumer = KafkaConsumer(
        "ai-detector",
        bootstrap_servers=[os.environ.get("KAFKA_IP")],
        auto_offset_reset='earliest',
        client_id="ai-detector-1",
        group_id=None,
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    BATCH_SIZE = 5

    for message in consumer:
        full_batch = message.value

        for i in range(0, len(full_batch), BATCH_SIZE):
            batch = full_batch[i:i+BATCH_SIZE]
            process_batch(batch, BATCH_SIZE)
