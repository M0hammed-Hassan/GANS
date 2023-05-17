# Libraries
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Current device = ',device)


# Hyperparameters
batch_size = 32
img_size = 64
channels = 3
channels_noise = 100
features_gen = 64
features_disc = 64
lr = 2e-4
betas = (0.5,0.999)
epochs = 100


# Preparing the data
transforms = transforms.Compose(
    [
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(channels)],[0.5 for _ in range(channels)])
    ]
)
dataset = torchvision.datasets.ImageFolder(root = 'data_sample',transform=transforms)
data_loader = torch.utils.data.DataLoader(dataset=dataset,shuffle=True,batch_size=batch_size)


# Building the nets.

class Discriminator(nn.Module):
    def __init__(self,in_channels,out_features):
        super(Discriminator,self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(in_channels = in_channels,out_channels = out_features,kernel_size=4,stride = 2,padding = 1),
            nn.LeakyReLU(0.2),
            self._block(out_features,out_features * 2,4,2,1),
            self._block(out_features * 2,out_features * 4,4,2,1),
            self._block(out_features * 4,out_features * 8,4,2,1),
            self._block(out_features * 8,1,4,2,0),
            nn.Sigmoid() 
        )
    def _block(self,in_channels,out_features,kernel_size,stride,padding):
        return nn.Sequential(
            nn.Conv2d(
                        in_channels,
                        out_features,
                        kernel_size,
                        stride,
                        padding,
                        bias = False 
                     ),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.2)
        )
    def forward(self,x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self,channels_noise,channels_imgs,feature_g):
        super(Generator,self).__init__()
        self.gen = nn.Sequential(
            self._block(channels_noise,feature_g  * 16,4,2,0),
            self._block(feature_g * 16,feature_g * 8,4,2,1),
            self._block(feature_g * 8,feature_g * 4,4,2,1),
            self._block(feature_g * 4,feature_g * 2,4,2,1),
            nn.ConvTranspose2d(feature_g * 2,channels_imgs,4,2,1),
            nn.Tanh()
        )
    def _block(self,in_channels,out_channles,kernel_size,stride,padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                        in_channels,
                        out_channles,
                        kernel_size,
                        stride,
                        padding,
                        bias = False
                     ),
            nn.BatchNorm2d(out_channles),
            nn.ReLU()
        )
    def forward(self,x):
        return self.gen(x)
    

# # # # # # # # # # # # # 
#                       #
#    Helper Function    #
#                       #   
# # # # # # # # # # # # # 

def init_weights(model):
    '''
    Initialize the modules weights into the models.
    '''
    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d,nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data,0.0,0.02) # std 0.02, mean 0.0

# Take objects
discriminator = Discriminator(channels,features_disc).to(device)
generator = Generator(channels_noise,channels,features_gen).to(device)

# Init the weights
init_weights(discriminator)
init_weights(generator)

# The optimizers
dis_opt = torch.optim.Adam(discriminator.parameters(),lr = lr,betas = betas)
gen_opt = torch.optim.Adam(generator.parameters(),lr = lr,betas=betas)

# The loss
loss = nn.BCELoss()


# Training process
for epoch in range(epochs):
    for idx,(imgs,_) in enumerate(data_loader):
        real_imgs = imgs.to(device)
        real_labels = torch.ones((len(real_imgs),1)).to(device)

        fake_imgs = torch.randn((len(real_imgs),channels_noise,1,1))
        fake_labels = torch.zeros((len(real_imgs),1))

        gen_fake_imgs = generator(fake_imgs)

        all_imgs_sample = torch.cat((real_imgs,gen_fake_imgs))
        all_labels_sample = torch.cat((real_labels,fake_labels))

        discriminator.zero_grad()
        disc_pred = discriminator(all_imgs_sample).reshape(all_labels_sample.shape)
        disc_loss = loss(disc_pred,all_labels_sample)
        disc_loss.backward()
        dis_opt.step()

        generator.zero_grad()
        gen_fake_imgs_pred = generator(fake_imgs)
        disc_fake_imgs_pred = discriminator(gen_fake_imgs_pred).reshape(real_labels.shape)
        gen_loss = loss(disc_fake_imgs_pred,real_labels)
        gen_loss.backward()
        gen_opt.step()

        if idx == 15:
            print(f'Epoch {epoch + 1}/{epochs}: disc_loss = {disc_loss:.3f} and gen_loss = {gen_loss:.3f}')