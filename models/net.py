import torch
import torch.nn as nn
from torchvision import models

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # self.encoder = nn.Sequential(*list(models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.children())[:-1])
        self.encoder = nn.Sequential(*list(models.vgg16().features.children())[:-1])
        self.decoder1 = nn.Sequential(nn.Conv2d(512,512,3,1,1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512,512,3,1,1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512,512,3,1,1),
                                      nn.ReLU(inplace=True),
                                      nn.ConvTranspose2d(512,512,2,2,0))
        
        self.decoder2 = nn.Sequential(nn.Conv2d(512,512,3,1,1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512,512,3,1,1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512,512,3,1,1),
                                      nn.ReLU(inplace=True),
                                      nn.ConvTranspose2d(512,512,2,2,0))
        
        self.decoder3 = nn.Sequential(nn.Conv2d(512,256,3,1,1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256,256,3,1,1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256,256,3,1,1),
                                      nn.ReLU(inplace=True),
                                      nn.ConvTranspose2d(256,256,2,2,0))
        
        self.decoder4 = nn.Sequential(nn.Conv2d(256,128,3,1,1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128,128,3,1,1),
                                      nn.ReLU(inplace=True),
                                      nn.ConvTranspose2d(128,128,2,2,0))
        
        self.decoder5 = nn.Sequential(nn.Conv2d(128,64,3,1,1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(64,64,3,1,1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(64,1,1,1,0),
                                      nn.Sigmoid())
        
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.decoder4(x)
        x = self.decoder5(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(4,3,1,1,1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(3,32,3,1,1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2,2,0),
                                      nn.Conv2d(32,64,3,1,1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(64,64,3,1,1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2,2,0),
                                      nn.Conv2d(64,64,3,1,1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(64,64,3,1,1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2,2,0))
        self.classifier = nn.Sequential(nn.Linear(64*28*28,100),
                                        nn.Tanh(),
                                        nn.Linear(100,2),
                                        nn.Tanh(),
                                        nn.Linear(2,1),
                                        nn.Sigmoid())

    
    def forward(self,x,c):
        x = torch.cat([x,c],dim=1)
        x = self.features(x)
        x = x.view(size=(x.shape[0],-1))
        x = self.classifier(x)
        return x




if __name__ == "__main__":
    g = Generator()
    d = Discriminator()
    x = torch.randn(size=(4,3,224,224))
    real_c = torch.randn(size=(4,1,224,224))
    fake_c = g(x)
    print(fake_c.shape)
    y1 = d(x,fake_c)
    y2 = d(x,real_c)
    print(y2.shape)
    



