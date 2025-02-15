import os
import torch
from models import Generator,Discriminator
from datasets import get_dataloader
from torchvision import transforms
from torchvision.utils import make_grid
import numpy as np
from PIL import Image
from tqdm import tqdm


def get_real_c(x,classifier_nets,device):
    x = transforms.Resize((224,224))(x)
    heats = torch.zeros(size=(x.shape[0],1,224,224),device=device)
    for net_type in classifier_nets.keys():
        net = classifier_nets[net_type]
        with torch.no_grad():
            pre = net(x).argmax(dim=1)
            f_map = net.features(x)
            f_map_reshape = f_map.reshape(f_map.shape[0], f_map.shape[1], -1)
            weights = net.classifier[-1].weight.data
            weights = torch.softmax(weights,dim=1)
            weights = weights[pre].unsqueeze(1)
            heat = torch.bmm(weights,f_map_reshape)
            heat = heat.reshape(f_map.shape[0],f_map.shape[-1], f_map.shape[-1]).unsqueeze(1)
            heat = torch.nn.functional.interpolate(heat, size=(224, 224), mode="bilinear", align_corners=False)
            min_val = heat.min(dim=2, keepdim=True)[0].min(dim=3,keepdim=True)[0]
            max_val = heat.max(dim=2, keepdim=True)[0].max(dim=3,keepdim=True)[0]
            heats += (heat - min_val) / (max_val - min_val)
    return heats / len(classifier_nets)



def vis(real_c,fake_c,epoch):
    real_c_grid = make_grid(real_c[:24].cpu()).numpy()
    Image.fromarray(np.array(real_c_grid[0] * 255,dtype=np.uint8)).save(f"temp/{epoch}_real_c.png")
    fake_c_grid = make_grid(fake_c[:24].cpu()).numpy()
    Image.fromarray(np.array(fake_c_grid[0] * 255,dtype=np.uint8)).save(f"temp/{epoch}_fake_c.png")



def train(net_G,net_D,dataloader,criterion,optimizer_G,optimizer_D,device,epochs,classifier_nets):
    bs = dataloader["train"].batch_size
    f = 0
    for epoch in range(1,epochs+1):
        net_G.train()
        net_D.train()
        for i,(x,_) in enumerate(dataloader["train"]):
            x = x.to(device)
            batch_size = x.shape[0]
            real_c = get_real_c(x,classifier_nets,device)
            # 训练判别器
            optimizer_D.zero_grad()
            # 计算判别器对真实图像的损失
            real_labels = torch.ones(batch_size,1,device=device)
            real_output = net_D(x,real_c)
            d_real_loss = criterion(real_output,real_labels)

            # 预测saliency map
            fake_c = net_G(x)
            # 计算判别器对预测图像的损失
            fake_labels = torch.zeros(batch_size, 1,device=device)
            fake_output = net_D(x,fake_c)
            d_fake_loss = criterion(fake_output, fake_labels)

            # 总判别损失
            d_loss= d_real_loss + d_fake_loss
            d_loss.backward()
            optimizer_D.step()

            # 训练生成器
            optimizer_G.zero_grad()
            fake_c = net_G(x)
            # 计算二元交叉熵损失
            bce_loss = criterion(fake_c,real_c)
            fake_labels = torch.ones(batch_size, 1,device=device)
            fake_output = net_D(x,fake_c)
            g_fake_loss = criterion(fake_output, fake_labels)
            g_loss = bce_loss + g_fake_loss
            g_loss.backward()
            optimizer_G.step()
            if i == 1 or i % 50 == 0:
                print(f"[{epoch}/{epochs}][{i+1}/{len(dataloader['train'])}][d_loss: {d_loss.item()} g_loss: {g_loss.item()}]")
            if f == 0 or f % 500 == 0:
                vis(real_c,fake_c,f)
            f += 1

def main():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    net_G,net_D = Generator().to(device),Discriminator().to(device)
    dataloader = get_dataloader(batch_size=32)
    criterion = torch.nn.BCELoss()
    optimizer_G = torch.optim.Adagrad(net_G.parameters(),lr=3e-4,weight_decay=1e-4)
    optimizer_D = torch.optim.Adagrad(net_D.parameters(),lr=3e-4,weight_decay=1e-4)
    epochs = 5
    classifier_nets = {}
    net = torch.load("ckpts/alexnet-GAP_best_acc_0.902.pth",map_location="cpu")
    net.eval()
    net.to(device)
    classifier_nets["alexnet-GAP"] = net
    train(net_G,net_D,dataloader,criterion,optimizer_G,optimizer_D,device,epochs,classifier_nets)


if __name__ == "__main__":
    main()

