"""
# File       : sketch_control.py
# Time       : 2023/6/29 15:00
# Author     : czw
# software   : PyCharm , Python3.7
# Description: 
"""
import os
import random

from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from cldm.ddim_hacked import DDIMSampler

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torchvision.transforms as T
import math
import argparse
import numpy as np
import einops
from PIL import Image
from cldm.model import create_model, load_state_dict
from annotator.util import resize_image, HWC3


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def get_views(panorama_height, panorama_width, window_size=64, stride=8):
    panorama_height /= 8
    panorama_width /= 8
    num_blocks_height = (panorama_height - window_size) // stride + 1
    num_blocks_width = (panorama_width - window_size) // stride + 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride)
        h_end = h_start + window_size
        w_start = int((i % num_blocks_width) * stride)
        w_end = w_start + window_size
        views.append((h_start, h_end, w_start, w_end))
    return views


class ControlNet(nn.Module):
    def __init__(self, device, cfg_path, ckpt_path):
        super().__init__()

        self.device = device
        self.model = create_model(cfg_path).cpu()
        self.model.load_state_dict(load_state_dict(ckpt_path, location='cuda'))
        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)
        self.scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder='scheduler')

        print(f'[INFO] loaded ControlNet stable diffusion!')

    @torch.no_grad()
    def get_random_background(self, n_samples):
        # sample random background with a constant rgb value
        backgrounds = torch.rand(n_samples, 3, device=self.device)[:, :, None, None].repeat(1, 1, 512, 512)
        return torch.cat([self.encode_imgs(bg.unsqueeze(0)) for bg in backgrounds])

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length',
                                      max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def encode_imgs(self, imgs):
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215
        return latents

    @torch.no_grad()
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    @torch.no_grad()
    def generate(self, sketches, masks, prompts, negative_prompts, height=512, width=512, num_inference_step=50,
                 guidance_scale=9, control_strength=1, bootstrapping=20, guess_mode=False, batch_size=4, ablation=0):
        cond = {'c_concat': [sketches], 'c_crossattn': [self.model.get_learned_conditioning(prompts)]}
        un_cond = {'c_concat': None if guess_mode else [sketches], 'c_crossattn': [self.model.get_learned_conditioning(negative_prompts)]}

        self.model.control_scales = [control_strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([control_strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

        b = len(prompts)


        latent = torch.randn((1, 4, height // 8, width // 8), device=self.device)
        noise = latent.clone().repeat(len(prompts) - 1, 1, 1, 1)
        count = torch.zeros_like(latent)
        value = torch.zeros_like(latent)

        self.ddim_sampler.make_schedule(ddim_num_steps=num_inference_step,
                                        ddim_eta=0.0,
                                        verbose=False)
        timesteps = make_ddim_timesteps(ddim_discr_method='uniform',
                                        num_ddim_timesteps=num_inference_step,
                                        num_ddpm_timesteps=1000,
                                        verbose=False)
        intermediates = {'x_inter': [latent], 'pred_x0': [latent]}
        time_range = np.flip(timesteps)
        print(f"Running DDIM Sampling with {num_inference_step} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=num_inference_step)
        bootstrapping_bg = torch.rand(bootstrapping, 4, device=self.device)[:, :, None, None].repeat(1, 1, 64, 64)
        imgs = []

        for i, step in enumerate(iterator):
            count.zero_()
            value.zero_()

            index = num_inference_step - i - 1
            ts = torch.full((b,), step, device=self.device, dtype=torch.long)
            latent_process = latent.repeat(len(prompts), 1, 1, 1)
            if i < bootstrapping:
                if ablation == 0:
                    # origin
                    latent_process[1:] = latent_process[1:] * masks[1:] + latent_process[0] * masks[0]
                elif ablation == 1:
                    # abalation1  no background
                    latent_process[1:] = latent_process[1:] * masks[1:]
                elif ablation == 2:
                    # abalation2  random color bg
                    bg = bootstrapping_bg[torch.randint(0, bootstrapping, (len(prompts) - 1,))]
                    bg = self.scheduler.add_noise(bg, noise, ts[0])
                    latent_process[1:] = latent_process[1:] * masks[1:] + bg * (1 - masks[1:])
            # latent_process[1:] = latent_process[1:] * masks[1:]
            batch_num = math.ceil(len(prompts) / batch_size)
            outs = None
            for j in range(batch_num):
                tmp_latent = latent_process[j * batch_size : (j + 1) * batch_size, :, :, :]

                tmp_cond = {}
                tmp_cond['c_concat'] = [cond['c_concat'][0][j * batch_size : (j + 1) * batch_size, :, :, :]]
                tmp_cond['c_crossattn'] = [cond['c_crossattn'][0][j * batch_size : (j + 1) * batch_size, :, :]]

                tmp_ts = ts[j * batch_size : (j + 1) * batch_size]

                tmp_un_cond = {}
                tmp_un_cond['c_concat'] = [un_cond['c_concat'][0][j * batch_size : (j + 1) * batch_size, :, :, :]]
                tmp_un_cond['c_crossattn'] = [un_cond['c_crossattn'][0][j * batch_size : (j + 1) * batch_size, :, :]]

                tmp_out = self.ddim_sampler.p_sample_ddim(x=tmp_latent,
                                                   c=tmp_cond,
                                                   t=tmp_ts,
                                                   index=index,
                                                   use_original_steps=False,
                                                   quantize_denoised=False,
                                                   temperature=1.0,
                                                   noise_dropout=0.0,
                                                   score_corrector=None,
                                                   corrector_kwargs=None,
                                                   unconditional_guidance_scale=guidance_scale,
                                                   unconditional_conditioning=tmp_un_cond,
                                                   dynamic_threshold=None)
                if outs == None:
                    outs = tmp_out[0].clone()
                else:
                    outs = torch.cat((outs, tmp_out[0].clone()), dim=0)

            latent_process = outs * masks
            value += latent_process.sum(dim=0, keepdims=True)
            count += masks.sum(dim=0, keepdims=True)
            latent = torch.where(count > 0, value / count, value)

            # tmp = self.model.decode_first_stage(latent)
            # tmp = (einops.rearrange(tmp, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            # tmp = [tmp[i] for i in range(len(tmp))]
            # tmp = Image.fromarray(tmp[0])
            # imgs.append(tmp)


        samples = self.model.decode_first_stage(latent)
        samples = (einops.rearrange(samples, 'b c h w -> b h w c') *127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        samples = [samples[i] for i in range(len(samples))]
        img = Image.fromarray(samples[0])

        return img, imgs


def preprocess_mask(mask_paths, h, w, device):
    bg_mask = torch.ones((1, 1, 512, 512)).to(device)
    masks = []
    for i in range(len(mask_paths)):
        mask_path = mask_paths[i]
        mask = np.array(Image.open(mask_path).convert("L").resize((512, 512)))
        mask = mask.astype(np.float32) / 255.0
        mask = mask[None, None]
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask).to(device)
        bg_mask -= mask
        mask = torch.nn.functional.interpolate(mask, size=(h, w), mode='nearest')
        masks.append(mask)
    bg_mask[bg_mask < 0] = 0
    bg_mask_resize = torch.nn.functional.interpolate(bg_mask, size=(h, w), mode='nearest')
    masks = [bg_mask_resize] + masks
    masks = torch.cat(masks, dim=0)
    return masks, bg_mask

def preprocess_sketch(sketch_path, h, w, device):
    sketch = np.array(Image.open(sketch_path).convert('RGB').resize((h, w)))
    detected_map = np.zeros_like(sketch, dtype=np.uint8)
    detected_map[np.min(sketch, axis=2) < 127] = 255
    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0

    return control

def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bg_negative', default='artifacts, text, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image, watermark, text, water, pool', type=str)  # 'artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image'
    parser.add_argument('--fg_negative', default=[''], type=list)  # 'artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image'
    parser.add_argument('--H', type=int, default=512)
    parser.add_argument('--W', type=int, default=512)
    parser.add_argument('--cfg_path', type=str, default='./models/cldm_v15.yaml')
    parser.add_argument('--ckpt_path', type=str, default='./models/control_sd15_scribble.pth')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--bootstrapping', type=int, default=20)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--sample_num', type=int, default=100)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--ablation', type=int, default=0, help='0:origin  1:no bg  2:random color bg')
    parser.add_argument('--process_list', type=int, default=0)
    parser.add_argument('--input_path', default='./inputs/test_example_01', type=str)
    parser.add_argument('--input_folder', default='./inputsAll/sketchyCOCOAblation', type=str)
    # parser.add_argument('--input_path', default='000000475423', type=str)
    opt = parser.parse_args()

    if opt.seed == -1:
        opt.seed = random.randint(0, 100000000)
    seed_everything(opt.seed)

    device = torch.device('cuda')

    ConN = ControlNet(device, opt.cfg_path, opt.ckpt_path)
    if opt.process_list == 0:
        input_path_list = [opt.input_path];
    else:
        input_folder = opt.input_folder
        input_path_list = os.listdir(input_folder)
        opt.sample_num = 1
        for i in range(len(input_path_list)):
            input_path_list[i] = os.path.join(input_folder, input_path_list[i])

    for u in range(len(input_path_list)):
        print('%d   /   %d    %s' % (u, len(input_path_list), input_path_list[u]))

        input_path = input_path_list[u]

        out_path = os.path.join(input_path, 'output')
        if opt.ablation == 1:
            out_path += '1'
        elif opt.ablation == 2:
            out_path += '2'


        if not os.path.exists(out_path):
            os.makedirs(out_path)

        if len(os.listdir(out_path)) >= 1 and opt.process_list == 1:
            print(out_path + 'skipped!')
            continue

        # 处理mask
        mask_paths = sorted(os.listdir(os.path.join(input_path, 'mask')))
        for i in range(len(mask_paths)):
            mask_paths[i] = os.path.join(input_path, 'mask', mask_paths[i])
        masks, bg_mask = preprocess_mask(mask_paths, opt.H // 8, opt.W // 8, device)


        # 处理草图
        sketch_paths = sorted(os.listdir(os.path.join(input_path, 'sketch')))
        for i in range(len(sketch_paths)):
            sketch_paths[i] = os.path.join(input_path, 'sketch', sketch_paths[i])

        bg_all_zero = torch.zeros((1, 3, opt.H, opt.W)).cuda()

        sketches = torch.stack([preprocess_sketch(sketch_path, opt.H, opt.W, device) for sketch_path in sketch_paths])
        control = einops.rearrange(sketches, 'b h w c -> b c h w').clone()
        # control = torch.cat((bg_all_zero, control), dim=0)

        # 处理背景草图
        bg_sketch = os.path.join(input_path, 'scene.png')
        bg = preprocess_sketch(bg_sketch, opt.H, opt.W, device).unsqueeze(0)
        bg_control = einops.rearrange(bg, 'b h w c -> b c h w').clone()
        bg_control = bg_control * bg_mask.squeeze(0).squeeze(0)

        # test: 输出mask后的背景草图
        fg_mask_output = (einops.rearrange(control, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(
            np.uint8)
        fg_mask_output = [fg_mask_output[i] for i in range(len(fg_mask_output))]
        img_test = Image.fromarray(fg_mask_output[0])
        img_test.save(os.path.join(input_path, 'test2.png'))

        bg_mask_output = (einops.rearrange(bg_control, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(
            np.uint8)
        bg_mask_output = [bg_mask_output[i] for i in range(len(bg_mask_output))]
        img_test = Image.fromarray(bg_mask_output[0]).convert('L')
        img_test.save(os.path.join(input_path, 'test.png'))


        # 把背景草图和前景草图cat起来
        control = torch.cat((bg_control, control), dim=0)


        # 处理文本引导
        with open(os.path.join(input_path, 'prompt.txt')) as f:
            prompts = f.readlines()
        for i in range(len(prompts)):
            prompts[i] += ', a photo, best quality, clear, extremely detailed, realistic, realistic photos'
        opt.fg_negative = ['artifacts, blurry, bad quality, distortions, unrealistic, distorted image, watermark, text'] * (len(prompts) - 1)
        neg_prompts = [opt.bg_negative] + opt.fg_negative

        shape = (4, opt.H // 8, opt.W // 8)

        for i in range(opt.sample_num):
            print('Sampleing the %d/%d img...' % (i, opt.sample_num))
            img, imgs = ConN.generate(sketches=control,
                                masks=masks,
                                prompts=prompts,
                                negative_prompts=neg_prompts,
                                height=opt.H,
                                width=opt.W,
                                num_inference_step=opt.steps,
                                guidance_scale=9,
                                control_strength=1,
                                bootstrapping=opt.bootstrapping,
                                guess_mode=False,
                                batch_size=opt.batch_size,
                                ablation=opt.ablation)
            # save image
            if opt.ablation == 0:
                img.save(os.path.join(input_path, 'output', '%d.png' % opt.seed))
                if len(imgs) > 0:
                    for j in range(len(imgs)):
                        imgs[j].save(os.path.join(input_path, 'output', '%d_%d.png' % (opt.seed, j)))
                print('image %s/output/%s.png saved' % (input_path, opt.seed))
            else:
                if not os.path.exists(os.path.join(input_path, 'output%d' % opt.ablation)):
                    os.makedirs(os.path.join(input_path, 'output%d' % opt.ablation))
                img.save(os.path.join(input_path, 'output%d' % opt.ablation, '%d.png' % opt.seed))
                print('image %s/output%d/%s.png saved' % (input_path, opt.ablation, opt.seed))
            opt.seed += 1
            seed_everything(opt.seed)
