import json
import os
from kafka import KafkaConsumer
from get_gpt_answer import GetGPTAnswer
from typing import List
from concurrent.futures import ThreadPoolExecutor
from predict_custom_model import predict_custom_trained_model
from google.protobuf.json_format import MessageToDict


def get_gpt_responses(data: dict[str, any], gpt_helper: GetGPTAnswer):
    data["gpt35_answer"] = gpt_helper.generate_gpt35_answer(data["question"])
    data["gpt4_answer"] = gpt_helper.generate_gpt4_answer(data["question"])
    return data


def process_batch(batch: List[dict[str, any]], batch_size: int, job_application_id: int):
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        gpt_helper = GetGPTAnswer()
        futures = [executor.submit(
            get_gpt_responses, data, gpt_helper) for data in batch]
        results = [future.result() for future in futures]

    print("Batch ready with gpt responses", results)

    predictions = predict_custom_trained_model(
        instances=results, project=os.environ.get("PROJECT_ID"), endpoint_id=os.environ.get("ENDPOINT_ID"))

    results = []
    for prediction in predictions:
        result_dict = {}
        for key, value in prediction._pb.items():
            # Ensure that 'value' is a protobuf message
            if hasattr(value, 'DESCRIPTOR'):
                result_dict[key] = MessageToDict(value)
            else:
                print(f"Item {key} is not a convertible protobuf message.")
        results.append(result_dict)

    print({"result": results})


def consume_messages():
    consumer = KafkaConsumer(
        "ai-detector",
        bootstrap_servers=[os.environ.get("KAFKA_IP")],
        auto_offset_reset='earliest',
        client_id="ai-detector-1",
        group_id=None,
    )

    print("Successfully connected to Kafka at", os.environ.get("KAFKA_IP"))

    BATCH_SIZE = 5

    for message in consumer:
        try:
            incoming_message = json.loads(message.value.decode("utf-8"))
            full_batch = incoming_message["data"]
        except json.JSONDecodeError:
            print("Failed to decode JSON from message:", message.value)
            print("Continuing...")
            continue

        for i in range(0, len(full_batch), BATCH_SIZE):
            batch = full_batch[i:i+BATCH_SIZE]
            process_batch(batch, BATCH_SIZE,
                          incoming_message["job_application_id"])
