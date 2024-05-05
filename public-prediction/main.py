from kafka_consumer import consume_messages
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    consume_messages()
