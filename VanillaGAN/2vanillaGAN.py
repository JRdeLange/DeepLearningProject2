import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
import numpy as np
import matplotlib
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm

# website: https://github.com/Susheel-1999/vanillaGAN-image_generation_using_pytorch/blob/main/Image_generation.ipynb

class Generator(nn.Module):
    def __init__(self, noise_vec):
        super(Generator,self).__init__()
        
        #size of noise vector
        self.noise_vec = noise_vec

        self.layer =nn.Sequential(
            #layer 1- linear layer 
            nn.Linear(self.noise_vec, 256),
            #activation function
            nn.LeakyReLU(),
            #layer 2- linear layer
            nn.Linear(256,512),
            #activation function
            nn.LeakyReLU(),

            #layer 3- linear layer
            nn.Linear(512,1024),
            #activation function
            nn.LeakyReLU(),

            #layer 4- linear layer
            nn.Linear(1024,784),
            #activation function
            nn.Tanh(),
        )
    
    #forward pass
    def forward(self, x):
        #forward pass of the generator
        forward_layer = self.layer(x)
        #reshaping
        #forward_layer = forward_layer.view(-1, 1, 28, 28)
        return forward_layer

#discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        #input dimension for discriminator
        self.input_dim = 784

        self.layer = nn.Sequential(
                #layer 1- linear layer 
                nn.Linear(self.input_dim,1024),
                #activation function
                nn.LeakyReLU(),
                #randomly zeroes some of the elements of the input tensor with probability 0.3
                nn.Dropout(0.3),
                
                #layer 2- linear layer 
                nn.Linear(1024,512),
                #activation function
                nn.LeakyReLU(),
                #randomly zeroes some of the elements of the input tensor with probability 0.3
                nn.Dropout(0.3),
                
                #layer 3- linear layer 
                nn.Linear(512,256),
                #activation function
                nn.LeakyReLU(),
                #randomly zeroes some of the elements of the input tensor with probability 0.3
                nn.Dropout(0.3),
                
                #layer 4- linear layer
                nn.Linear(256,1),
                #activation function
                nn.Sigmoid(),
        )
    
    #forward pass
    def forward(self,x):
        #reshaping
        #x = x.view(-1, 784)
        #forward pass of the discriminator
        forward_layer = self.layer(x)
        return  forward_layer

def train_generator(fake_data, real_labels, fake_labels):
	generator_opt.zero_grad()
	output_fake = discriminator(fake_data)
	loss = criterion(output_fake, fake_labels)
	loss.backward
	generator_opt.step()
	return loss

def train_discriminator(real_data, fake_data, real_labels, fake_labels):
	discriminator_opt.zero_grad()
	output_real = discriminator(real_data)
	loss_real = criterion(output_real, real_labels)

	output_fake = discriminator(fake_data)
	loss_fake = criterion(output_fake, fake_labels)

	loss_real.backward()
	loss_fake.backward()
	discriminator_opt.step()
	loss = loss_real + loss_fake
	return loss


batch_size = 64 
noise_vec = 128 # size of the noise vector for the generator
dis_steps = 1 # number of steps applied to the discriminator

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_noise(sample_size, noise_vec):
	return torch.randn(sample_size, noise_vec).to(DEVICE)

transform = transforms.Compose([
	transforms.Resize((256,256)),
	transforms.ToTensor(), 
	transforms.Normalize((0.5,),(0.5,)),])

root_path = "archive/front/"
data = datasets.ImageFolder(root = root_path, transform=transform)
data_loader = DataLoader(dataset = data, batch_size = batch_size, shuffle = True)

generator = Generator(noise_vec).to(DEVICE)
discriminator = Discriminator().to(DEVICE)

criterion = nn.BCELoss()
discriminator_opt = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
generator_opt = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

real_labels = torch.ones(batch_size, 1).to(DEVICE)
fake_labels = torch.zeros(batch_size, 1).to(DEVICE)

generator.train()
discriminator.train()

losses_generator = []
losses_discriminator = []
images = []

epochs = 15

for epoch in range(epochs):

	loss_generator = 0.0
	loss_discriminator = 0.0
	for bi, dat in tqdm(enumerate(data_loader), total=int(len(data)/data_loader.batch_size)):
		image,_=dat
		image = image.to(DEVICE)
		size = len(image)
		for step in range(dis_steps):
			fake_data = generator(create_noise(size, noise_vec)).detach()
			real_data = image
			loss_discriminator = loss_discriminator + train_discriminator(real_data, fake_data, real_labels, fake_labels)

			fake_data = generator(create_noise(size, noise_vec))
			loss_generator = loss_generator + train_generator(fake_data, real_labels, fake_labels)

		epoch_loss_gen = loss_generator / bi
		epoch_loss_dis = loss_discriminator / bi
		losses_generator.append(epoch_loss_gen)
		losses_discriminator.append(epoch_loss_dis)

		print("Epoch: ", epoch)
		print("Generator loss :", epoch_loss_gen, "Discriminator loss :",epoch_loss_dis)

print('Generator is ready to generate images! Saving the model..')
torch.save(generator.state_dict(), 'model/generator.pth')

sample_size = 64 
noise = create_noise(sample_size, noise_vec)
generator = generator(noise_vec)


#load the trained model
generator.load_state_dict(torch.load("model/generator.pth"))
generator.eval()
generator.to(DEVICE)

generated_img=generator(noise).detach()
#save the generated image
generated_img = make_grid(generated_img)
save_image(generated_img,"output_img/gen_img.png")

