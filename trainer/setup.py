from abc import ABCMeta, abstractmethod

from trainer.trainer_props import TrainerPropsBase


class Setup(metaclass=ABCMeta):

    @abstractmethod
    def setup(self) -> dict:
        pass

    def __call__(self):
        return self.setup()
