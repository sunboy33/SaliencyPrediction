from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader,Subset




def get_dataloader(batch_size):
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

    v_transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
    

    t_dataset = CIFAR10(root="datasets",train=True,transform=transform,download=False)
    v_dataset = CIFAR10(root="datasets",train=False,transform=v_transform,download=False)

    # t_dataset = Subset(t_dataset,list(range(1024)))
    # v_dataset = Subset(v_dataset,list(range(1024,1500)))
    td = DataLoader(t_dataset,shuffle=True,batch_size=batch_size,num_workers=0)
    vd = DataLoader(v_dataset,shuffle=False,batch_size=batch_size,num_workers=0)
    return {"train":td,"val":vd}

