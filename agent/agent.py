from abc import abstractmethod, ABCMeta

class AgentBase(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def update_policy(self, input_data):
        pass

    @abstractmethod
    def prediction(self, input_data):
        pass
