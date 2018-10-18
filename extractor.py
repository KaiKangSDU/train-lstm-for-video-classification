import torch.nn as nn
import torch
from VGG_Face_torch import VGG_Face_torch\





class VGG_Net(nn.Module):
    def __init__(self, model):
        super(VGG_Net, self).__init__()

        self.pre_model = nn.Sequential(*list(model.children())[:-1])
        # self.dropout = nn.Dropout(p=0.8)
        self.classifier = nn.Linear(4096, 7)

    def forward(self, x):
        x = self.pre_model(x)
        # x = self.dropout(x)
        x = self.classifier(x)

        return x

model_emotion = VGG_Face_torch
model = VGG_Net(model_emotion)
if torch.cuda.is_available():
    model = VGG_Net(model_emotion).cuda()
else:
    model = VGG_Net(model_emotion)















