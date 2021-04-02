from __future__ import print_function
import torch
from torch import nn 
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import os
import random
import torchvision.utils as vutils
from torchvision.utils import make_grid, save_image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from time import time 

image_size = 64

epochs = 100

batch_size = 12

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data 
root_path = "data/"

transform = transforms.Compose([
	transforms.Resize((image_size, image_size)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.5], std=[0.5])]
	)

data = datasets.ImageFolder(root = root_path, transform=transform)
data_loader = DataLoader(dataset = data, batch_size = batch_size, shuffle = True, drop_last = True)

#---------------------------------------------------------
#---------------------------------------------------------
#---------------------------------------------------------

class Generator(nn.Module):
	def __init__(self, latent_dim):
		super(Generator, self).__init__()
		output_dim = image_size * image_size * 3
		self.latent_dim = latent_dim 
		
		self.main = nn.Sequential(
			nn.Linear(latent_dim, 256),
			nn.LeakyReLU(),

			nn.Linear(256, 512),
			nn.LeakyReLU(),

			nn.Linear(512, 1024),
			nn.LeakyReLU(),

			nn.Linear(1024, output_dim),
			nn.Tanh()
			)
	def forward(self, input_tensor):
		input_tensor = input_tensor.view(input_tensor.size(0), -1)
		intermediate = self.main(input_tensor)
		return intermediate

#---------------------------------------------------------
#---------------------------------------------------------

class  Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()

		input_dim = image_size * image_size * 3
		output_dim = 1
		self.main = nn.Sequential(
			nn.Linear(input_dim, 1024),
			nn.LeakyReLU(),
			nn.Dropout(0.3),
						
			nn.Linear(1024, 512),
			nn.LeakyReLU(),
			nn.Dropout(0.3),
						
			nn.Linear(512, 256),
			nn.LeakyReLU(),
			nn.Dropout(0.3),
						
			nn.Linear(256, output_dim),
			nn.Sigmoid()
			)


	def forward(self, input_tensor):
		i = input_tensor.view(input_tensor.size(0), -1)
		#print(i.size())
		#input_tensor = input_tensor.view(batch_size, -1)
		#print(input_tensor.size())
		intermediate = self.main(i)
		return intermediate


#---------------------------------------------------------
#---------------------------------------------------------
#---------------------------------------------------------


class VanillaGAN():
	def __init__(self, generator, discriminator, lr_d=1e-3, lr_g=2e-4):
		self.generator = generator.to(DEVICE)
		self.discriminator = discriminator.to(DEVICE)
		self.criterion = nn.BCELoss()
		self.optim_d = optim.Adam(discriminator.parameters(), lr = lr_d, betas=(0.5, 0.999))
		self.optim_g = optim.Adam(generator.parameters(), lr = lr_g, betas=(0.5,0.999))
		self.target_ones = torch.ones((batch_size, 1)).to(DEVICE)
		self.target_zeros = torch.zeros((batch_size, 1)).to(DEVICE)

	def generate_samples(self, latent_vec=None, num=None):
		num = batch_size if num is None else num
		latent_vec = create_noise(num) if latent_vec is None else latent_vec
		with torch.no_grad():
			samples = self.generator(latent_vec)
		return samples

	def train_step_generator(self):
		self.generator.zero_grad()

		latent_vec = create_noise(batch_size)
		generated = self.generator(latent_vec)
		classifications = self.discriminator(generated)
		loss = self.criterion(classifications, self.target_ones)
		loss.backward()
		self.optim_g.step()
		return loss.item()

	def train_step_discriminator(self, images):
		self.discriminator.zero_grad()
		#print(images.size())
		real_samples = images.to(DEVICE)
		#print(real_samples.size())
		pred_real = self.discriminator(real_samples)
		loss_real = self.criterion(pred_real, self.target_ones)

		latent_vec = create_noise(batch_size)
		with torch.no_grad():
			fake_samples = self.generator(latent_vec)
		pred_fake = self.discriminator(fake_samples)
		loss_fake = self.criterion(pred_fake, self.target_zeros)

		loss = (loss_real + loss_fake) /2
		loss.backward()
		self.optim_d.step()
		return loss_real.item(), loss_fake.item()

	def train_step(self, images):
		loss_d = self.train_step_discriminator(images)
		loss_g = self.train_step_generator()
		return loss_g, loss_d

#---------------------------------------------------------
#---------------------------------------------------------
#---------------------------------------------------------
def create_noise(sample_size):
	return torch.randn(sample_size, 100).to(DEVICE)

generator = Generator(100).to(DEVICE)
discriminator = Discriminator().to(DEVICE)

gan = VanillaGAN(generator, discriminator)

loss_g, loss_d_real, loss_d_fake = [], [], []
start = time()
img_list = []
for epoch in range(epochs):

	loss_g_running, loss_d_real_running, loss_d_fake_running = 0, 0, 0
	
	for i, images in enumerate(data_loader, 0):
		lg_, (ldr_, ldf_) = gan.train_step(images[0])
		loss_g_running += lg_
		loss_d_real_running += ldr_
		loss_d_fake_running += ldf_

	loss_g.append(loss_g_running / batch_size)
	loss_d_real.append(loss_d_real_running / batch_size)
	loss_d_fake.append(loss_d_fake_running / batch_size)

	print(f"Epoch {epoch+1}/{epochs} ({int(time() - start)}s):"
			f" G={loss_g[-1]:.3f},"
			f" Dr={loss_d_real[-1]:.3f},"
			f" Df={loss_d_fake[-1]:.3f}")

	if (epoch % 500 == 0) or ((epoch == epochs-1) and (i == len(data_loader)-1)):
		fixed_noise = create_noise(batch_size)
		with torch.no_grad():
			fake = generator(fixed_noise).detach().cpu()
		img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

#---------------------------------------------------------
#---------------------------------------------------------
#---------------------------------------------------------	
generator.eval()
torch.save(generator.state_dict(), 'generator.pth')
print("generator saved.")

"""noise = create_noise(batch_size)
generated_img = generator(noise).detach()
generated_img = make_grid(generated_img)
save_image(generated_img, "images.png")"""

fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
plt.savefig("images1")
plt.show()

HTML(ani.to_jshtml())

plt.figure(figsize=(15,15))
plt.axis("off")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.savefig("images")
