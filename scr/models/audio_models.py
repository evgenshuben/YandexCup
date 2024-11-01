import torch
import torch.nn as nn
from torchvggish import vggish


class VGGishModel(nn.Module):
    def __init__(self, emb_size=128, num_classes=10, postprocess=False):
        super(VGGishModel, self).__init__()
        self.vggish = vggish(postprocess)
        self.emb_head = nn.Linear(128, emb_size)  # 128 - размер эмбеддинга VGGish
        self.cls_head = nn.Linear(emb_size, num_classes)  # num_classes - количество классов

    def forward(self, x):
        vggish_output = self.vggish(x)
        emb = self.emb_head(vggish_output)  # (N, emb_size)
        cls = self.cls_head(emb)  # (N, num_classes)
        return {"emb": emb, "cls": cls}