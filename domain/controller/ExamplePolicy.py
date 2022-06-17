from torch import nn
from .calc_size import calc_size
from domain.convert_data import to_tensor
from ..ImageObject import ImageObject

'''
・サンプルです
'''

class ExamplePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.channel = 3
        self.w       = 64
        self.h       = 64
        output_shape = 9

        self.c1 = nn.Conv2d(self.channel, 6, kernel_size=5, padding=2)
        self.r1 = nn.ReLU(inplace=True)
        self.m1 = nn.MaxPool2d(2)
        self.c2 = nn.Conv2d(6, 16, kernel_size=5)
        self.r2 = nn.ReLU(inplace=True)
        self.m1 = nn.MaxPool2d(2)
        self.f1 = nn.Flatten()
        self.l1 = nn.Linear(3136, 128)
        self.r3 = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(128, 64)
        self.r4 = nn.ReLU(inplace=True)
        self.l3 = nn.Linear(64, output_shape)
        self.s  = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, self.channel, self.w, self.h)  # ; print(xb.shape)
        x = self.c1(x)      # ; print(x.shape)
        x = self.r1(x)      # ; print(x.shape)
        x = self.m1(x)      # ; print(x.shape)
        x = self.c2(x)      # ; print(x.shape)
        x = self.r2(x)      # ; print(x.shape)
        x = self.m1(x)      # ; print(x.shape)
        x = self.f1(x)      # ; print(x.shape)
        x = self.l1(x)      # ; print(x.shape)
        x = self.r3(x)      # ; print(x.shape)
        x = self.l2(x)      # ; print(x.shape)
        x = self.r4(x)      # ; print(x.shape)
        x = self.l3(x)      # ; print(x.shape)
        x = self.s(x)
        return x.view(-1, x.size(1))


    def get_action(self, image: ImageObject):
        assert isinstance(image, ImageObject)
        state  = to_tensor(image.channel_first)         # pytorchのpolicyに入力するためにchannel_firstで取り出してtensorに変換
        action = self.forward(state)                    # policyからactionを取得
        action = action.detach().numpy().reshape(-1)    # tensorからnumpyに変換
        return action
