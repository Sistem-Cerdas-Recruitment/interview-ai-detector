from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


class GetGPTAnswer:
    def __init__(self):
        self.llm_gpt4o = ChatOpenAI(model="gpt-4o")

    def generate_gpt4o_answer(self, question: str):
        messages = [
            SystemMessage(
                content="Please answer the following question based solely on your internal knowledge, without external references. Assume you are the human."),
            HumanMessage(question)
        ]

        gpt4_answer = self.llm_gpt4o.invoke(messages)
        return gpt4_answer.content
