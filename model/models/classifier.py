import torch
import torch.nn as nn
import numpy as np
from model.utils import euclidean_metric
import torch.nn.functional as F
from model.models import FewShotModel
from torchvision import models
    
class Classifier(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone_class == 'ConvNet':
            from model.networks.convnet import ConvNet
            hdim = 64
            self.encoder = ConvNet()
        elif args.backbone_class == 'Res12':
            hdim = 640
            from model.networks.res12 import ResNet
            self.encoder = ResNet()
            #self.encoder = models.resnet18(pretrained=True)
        elif args.backbone_class == 'Res18':
            hdim = 512
            from model.networks.res18 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'WRN':
            hdim = 640
            from model.networks.WRN28 import Wide_ResNet
            self.encoder = Wide_ResNet(28, 10, 0.5)
        else:
            raise ValueError('')

        self.fc = nn.Linear(hdim, args.num_class)

    def forward(self, instance_embs):
        out = self.encoder(instance_embs)
        out = self.fc(out)
        return out

    def forward_proto(self, data_shot, data_query, way = None):
        if way is None:
            way = self.args.num_class
        proto = self.encoder(data_shot)
        proto = proto.reshape(self.args.shot, way, -1).mean(dim=0)
        query = self.encoder(data_query)
        
        logits_dist = euclidean_metric(query, proto)
        logits_sim = torch.mm(query, F.normalize(proto, p=2, dim=-1).t())
        return logits_dist, logits_sim
