import torch 
import torch.nn as nn
from torchvision import models


class Visionencoder(nn.Module):
    def __init__(self, embedding_dim =512):
        super().__init__()

        resnet= models.resnet50(weights=  models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone= nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Linear(2048, embedding_dim)

    def forward(self, x):

        x= self.backbone(x)
        x= x.view(x.size(0),-1)
        x= self.projection(x)
        return torch.nn.functional.normalize(x, p=2, dim=1) 

#test
model = Visionencoder(embedding_dim=512) 
x= torch.randn(4,3,224,224)
embeddings = model(x)
print(embeddings.shape)      