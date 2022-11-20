import torch
import numpy as np
from torch.nn import functional as F
import os
import math 



def conv_warper(layer, input, style, noise):
	# the conv should change
	conv = layer.conv
	batch, in_channel, height, width = input.shape

	style = style.view(batch, 1, in_channel, 1, 1)
	weight = conv.scale * conv.weight * style
	
	if conv.demodulate:
		demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
		weight = weight * demod.view(batch, conv.out_channel, 1, 1, 1)

	weight = weight.view(
		batch * conv.out_channel, in_channel, conv.kernel_size, conv.kernel_size
	)

	if conv.upsample:
		input = input.view(1, batch * in_channel, height, width)
		weight = weight.view(
			batch, conv.out_channel, in_channel, conv.kernel_size, conv.kernel_size
		)
		weight = weight.transpose(1, 2).reshape(
			batch * in_channel, conv.out_channel, conv.kernel_size, conv.kernel_size
		)
		out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
		_, _, height, width = out.shape
		out = out.view(batch, conv.out_channel, height, width)
		out = conv.blur(out)

	elif conv.downsample:
		input = conv.blur(input)
		_, _, height, width = input.shape
		input = input.view(1, batch * in_channel, height, width)
		out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
		_, _, height, width = out.shape
		out = out.view(batch, conv.out_channel, height, width)

	else:
		input = input.view(1, batch * in_channel, height, width)
		out = F.conv2d(input, weight, padding=conv.padding, groups=batch)
		_, _, height, width = out.shape
		out = out.view(batch, conv.out_channel, height, width)
		
	out = layer.noise(out, noise=noise)
	out = layer.activate(out)
	
	return out

def decoder(G, style_space, latent, noise, resize_image = True):
	# an decoder warper for G
	out = G.input(latent)
	out = conv_warper(G.conv1, out, style_space[0], noise[0])
	skip = G.to_rgb1(out, latent[:, 1])
	

	i = 1
	for conv1, conv2, noise1, noise2, to_rgb in zip(
		G.convs[::2], G.convs[1::2], noise[1::2], noise[2::2], G.to_rgbs
	):	
		
		out = conv_warper(conv1, out, style_space[i], noise=noise1)
		out = conv_warper(conv2, out, style_space[i+1], noise=noise2)
		skip = to_rgb(out, latent[:, i + 2], skip)	
		i += 2

	image = skip

	if resize_image:
		face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))	
		image = face_pool(image)
	return image

def encoder(G, noise, truncation, truncation_latent, size = 256, input_is_latent = False):
	style_space = []
	# an encoder warper for G
	inject_index = None
	if not input_is_latent:
		inject_index = G.n_latent
		styles = [noise]
		styles = [G.style(s) for s in styles]	
	else:
		styles = [noise]

	n_latent = int(math.log(size, 2))* 2 - 2
	if truncation < 1:
		style_t = []
		for style in styles:
			style_t.append(
				truncation_latent + truncation * (style - truncation_latent)
			)
		styles = style_t
	
	if len(styles) < 2:
		inject_index = n_latent
		if styles[0].ndim < 3:
			latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
			
		else:
			latent = styles[0]
	
	else:
		if inject_index is None:
			inject_index = random.randint(1, n_latent - 1)

		latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
		latent2 = styles[1].unsqueeze(1).repeat(1, n_latent - inject_index, 1)
		latent = torch.cat([latent, latent2], 1)

	noise = [getattr(G.noises, 'noise_{}'.format(i)) for i in range(G.num_layers)]
	
	style_space.append(G.conv1.conv.modulation(latent[:, 0]))
	i = 1
	for conv1, conv2, noise1, noise2, to_rgb in zip(
		G.convs[::2], G.convs[1::2], noise[1::2], noise[2::2], G.to_rgbs
	):	
		style_space.append(conv1.conv.modulation(latent[:, i]))
		style_space.append(conv2.conv.modulation(latent[:, i+1]))
		i += 2
	
	return style_space, latent, noise

