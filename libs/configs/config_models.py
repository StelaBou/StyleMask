import os
import numpy as np

stylegan2_ffhq_1024 = {
	'image_resolution':			1024,
	'channel_multiplier':		2,
	'gan_weights':				'./pretrained_models/stylegan2-ffhq-config-f_1024.pt',

	'stylespace_dim':			6048,
	'split_sections':			[512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 256, 256, 128, 128, 64, 64, 32],

	'e4e_inversion_model':		'./pretrained_models/e4e_ffhq_encode_1024.pt',
	'expression_ranges':		'./libs/configs/ranges_FFHQ.npy' # Used for evaluation
}

