from abc import ABCMeta, abstractmethod

'''
    ・環境の抽象クラスです
    ・具象クラスはこの抽象クラスを継承し，抽象クラス内で定義されているメソッドを実装する必要があります
    ・OpenAI gym などの特定の環境を再現しているわけではないので，一般的に公開されているもととは
    　少し異なっているかもしれません
'''

class AbstractEnvironment(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, config):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def set_ctrl(self):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def render(self):
        pass