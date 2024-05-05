from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


class GetGPTAnswer:
    def __init__(self):
        self.llm_gpt35 = ChatOpenAI(model="gpt-3.5-turbo")
        self.llm_gpt4 = ChatOpenAI(model="gpt-4-turbo")

    def generate_gpt35_answer(self, question: str):
        messages = [
            SystemMessage(
                content="Please answer the following question based solely on your internal knowledge, without external references. Assume you are the human."),
            HumanMessage(question)
        ]

        gpt35_answer = self.llm_gpt35.invoke(messages)
        return gpt35_answer.content

    def generate_gpt4_answer(self, question: str):
        messages = [
            SystemMessage(
                content="Please answer the following question based solely on your internal knowledge, without external references. Assume you are the human."),
            HumanMessage(question)
        ]

        gpt4_answer = self.llm_gpt4.invoke(messages)
        return gpt4_answer.content
