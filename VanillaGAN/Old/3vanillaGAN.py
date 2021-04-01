from __future__ import print_function
import torch
from torch import nn 
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import os
import random
#import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# wanna adapt it to this one a bit more https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f
# website: https://towardsdatascience.com/pytorch-and-gans-a-micro-tutorial-804855817a6b
# Dc only -> 64 by 64 pixels

class Generator(nn.Module):
	def __init__(self, latent_dim, ouput_actiation=None):
		super(Generator, self).__init__()
		self.main = nn.Sequential(
			nn.Linear(latent_dim, 64),
			nn.LeakyReLU(),
			nn.Linear(64, 32),
			nn.Linear(32, 1),
			nn.Tanh()
			)

	def forward(self, input_tensor):
		intermediate = self.main(input_tensor)
		return intermediate

class  Discriminator(nn.Module):
	def __init__(self, input_dim, layers):
		super(Discriminator, self).__init__()
		self.input_dim = input_dim

		self.module_list = nn.ModuleList()
		last_layer = self.input_dim
		for i, width in enumerate(layers):
			self.module_list.append(nn.Linear(last_layer, width))
			last_layer = width
			if (i+1) != len(layers):
				self.module_list.append(nn.LeakyReLU())
			else:
				self.module_list.append(nn.Sigmoid())

	def forward(self, input_tensor):
		intermediate = input_tensor
		for layer in self.module_list:
			intermediate = layer(intermediate)
		return intermediate


class VanillaGAN():
	def __init__(self, generator, discriminator, noise_fn, batch_size=32, device='cpu', lr_d=1e-3, lr_g=2e-4):
		self.generator = generator.to(device)
		self.discriminator = discriminator.to(device)
		self.noise_fn = noise_fn
		self.batch_size = batch_size
		self.device = device
		self.criterion = nn.BCELoss()
		self.optim_d = optim.Adam(discriminator.parameters(), lr = lr_d, betas=(0.5, 0.999))
		self.optim_g = optim.Adam(generator.parameters(), lr = lr_g, betas=(0.5,0.999))
		self.target_ones = torch.ones((batch_size, 1)).to(device)
		self.target_zeros = torch.zeros((batch_size, 1)).to(device)

	def generate_samples(self, latent_vec=None, num=None):
		num = self.batch_size if num is None else num
		latent_vec = self.noise_fn(num) if latent_vec is None else latent_vec
		with torch.no_grad():
			samples = self.generator(latent_vec)
		return samples

	def train_step_generator(self):
		self.generator.zero_grad()

		latent_vec = self.noise_fn(self.batch_size)
		generated = self.generator(latent_vec)
		classifications = self.discriminator(generated)
		loss = self.criterion(classifications, self.target_ones)
		loss.backward()
		self.optim_g.step()
		return loss.item()

	def train_step_discriminator(self, images):
		self.discriminator.zero_grad()

		real_samples = images.to(self.device)
		pred_real = self.discriminator(real_samples)
		loss_real = self.criterion(pred_real, self.target_ones)

		latent_vec = self.noise_fn(self.batch_size)
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


def main():
	from time import time 
	epochs = 15
	batches = 10
	generator = Generator(1)
	discriminator = Discriminator(1, [64, 32, 1])
	noise_fn = lambda x: torch.rand((x, 1), device='cpu')

	img_list = []

	transform = transforms.Compose([
		transforms.Resize((256,256)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5], std=[0.5])]
		)
	root_path = "archive/front/"
	data = datasets.ImageFolder(root = root_path, transform=transform)
	data_loader = DataLoader(dataset = data, batch_size = batches, shuffle = True, drop_last = False)
	gan = VanillaGAN(generator, discriminator, noise_fn, device='cpu')

	loss_g, loss_d_real, loss_d_fake = [], [], []
	start = time()
	for epoch in range(epochs):
		loss_g_running, loss_d_real_running, loss_d_fake_running = 0, 0, 0
		for i, images in enumerate(data_loader, 0):
			lg_, (ldr_, ldf_) = gan.train_step(images[0])
			oss_g_running += lg_
			loss_d_real_running += ldr_
			loss_d_fake_running += ldf_
		loss_g.append(loss_g_running / batches)
		loss_d_real.append(loss_d_real_running / batches)
		loss_d_fake.append(loss_d_fake_running / batches)

		print(f"Epoch {epoch+1}/{epochs} ({int(time() - start)}s):"
				f" G={loss_g[-1]:.3f},"
				f" Dr={loss_d_real[-1]:.3f},"
				f" Df={loss_d_fake[-1]:.3f}")
		if (epoch % 500 == 0) or ((epoch == epochs-1) and (i == len(dataloader)-1)):
			with torch.no_grad():
				fake = netG(fixed_noise).detach().cpu()
			img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

	generator.eval()
	torch.save(generator.state_dict(), 'model/generator.pth')
	fig = plt.figure(figsize=(8,8))
	plt.axis("off")
	ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
	ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
	HTML(ani.to_jshtml())



if __name__ == "__main__":
	main()
