# Libraries

import os
import torch
import torchvision
from torchvision.transforms import transforms
import torch.nn as nn
import matplotlib.pyplot as plt

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Data Preparation

os.makedirs('data')
transform = transforms.Compose(
    [
        transforms.ToTensor(), 
        transforms.Normalize(mean = [0.5],std = [0.5]) 
    ]
)
train_set = torchvision.datasets.MNIST(root = 'data',train = True,download = True,transform = transform)
batch_size = 32
train_loader = torch.utils.data.DataLoader(dataset = train_set,batch_size = batch_size,shuffle = True)



# Plot imags
imgs,labels = next(iter(train_loader))
for idx,img in enumerate(imgs):
  plt.subplot(4, 8, idx + 1)
  plt.imshow(img.reshape(28,28),cmap = 'gray')
  plt.title(labels[idx].item())
  plt.axis('off')
plt.show()


# Building GAN Nets

# Discriminator
class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator,self).__init__()
    self.disc = nn.Sequential(

        nn.Linear(784,512),
        nn.LeakyReLU(0.01),

        nn.Linear(512,256),
        nn.LeakyReLU(0.001),

        nn.Linear(256,128),
        nn.LeakyReLU(0.001),

        nn.Linear(128,1),
        nn.Sigmoid()
    )
  def forward(self,x):
    x = x.view(x.size(0),784) # 32*784
    res = self.disc(x)
    return res
  
# Generator
class Generator(nn.Module):
  def __init__(self):
    super(Generator,self).__init__()
    self.gen = nn.Sequential(
        
        nn.Linear(100,128),
        nn.LeakyReLU(0.001),

        nn.Linear(128,256),
        nn.LeakyReLU(0.001),

        nn.Linear(256,512),
        nn.LeakyReLU(0.001),

        nn.Linear(512,784),
        nn.Tanh() 
    )
  def forward(self,x):
    x = self.gen(x)
    out = x.view(x.size(0),1,28,28) 
    return out

# Obj.
discriminator = Discriminator().to(device)
generator = Generator().to(device)


# Hyperparameters
lr = 3e-4
epochs = 100

# Optimizers and loss func
loss = nn.BCELoss()
disc_opt = torch.optim.Adam(discriminator.parameters(),lr = lr)
gen_opt = torch.optim.Adam(generator.parameters(),lr = lr)

for epoch in range(epochs):
  for idx,(imgs,_) in enumerate(train_loader):
    real_imgs = imgs.to(device)
    real_labels = torch.ones((batch_size,1)).to(device)

    fake_imgs = torch.randn((batch_size,100)).to(device) # Random nums from normal dist.
    fake_labels = torch.zeros((batch_size,1)).to(device)

    gen_fake_imgs = generator(fake_imgs)

    all_sample = torch.cat((real_imgs,gen_fake_imgs))
    all_labels = torch.cat((real_labels,fake_labels))

    discriminator.zero_grad()
    disc_pred = discriminator(all_sample)
    disc_loss = loss(disc_pred,all_labels)
    disc_loss.backward()
    disc_opt.step()

    generator.zero_grad()
    gen_fake_imgs = generator(fake_imgs)
    disc_fake_pred = discriminator(gen_fake_imgs)
    gen_loss = loss(disc_fake_pred,real_labels)
    gen_loss.backward()
    gen_opt.step()

    if idx == (batch_size - 1):
      print(f'Epoch {epoch + 1 }/{epochs} res: dis_loss = {disc_loss:.3f}, gen_loss = {gen_loss:.3f}') 


# Test
fake_imgs_sample = torch.randn((batch_size,100)).to(device)
gen_pred = generator(fake_imgs_sample)
gen_pred = gen_pred.cpu().detach()
for idx,img in enumerate(gen_pred):
  plt.subplot(4, 8, idx + 1)
  plt.imshow(img.reshape(28,28,1),cmap = 'gray')
  plt.axis('off')
plt.show()





