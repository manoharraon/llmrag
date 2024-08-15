from langchain_ollama import ChatOllama

class ModelManager:
    def __init__(self):
        self.current_model = "llama3.1"
        self.llm = ChatOllama(model=self.current_model, temperature=0)

    def set_model(self, model_name):
        if model_name != self.current_model:
            self.current_model = model_name
            self.llm = ChatOllama(model=self.current_model, temperature=0)

    def get_llm(self):
        return self.llm