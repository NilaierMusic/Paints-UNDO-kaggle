import os
import argparse
import functools
import random
import numpy as np
import torch
import wd14tagger
import memory_management
import uuid
from PIL import Image
from diffusers_helper.code_cond import unet_add_coded_conds
from diffusers_helper.cat_cond import unet_add_concat_conds
from diffusers_helper.k_diffusion import KDiffusionSampler
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers_vdm.pipeline import LatentVideoDiffusionPipeline
from diffusers_vdm.utils import resize_and_center_crop, save_bcthw_as_mp4

# Set environment variables and directories
os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), 'hf_download')
result_dir = os.path.join('./', 'results')
os.makedirs(result_dir, exist_ok=True)

class ModifiedUNet(UNet2DConditionModel):
    @classmethod
    def from_config(cls, *args, **kwargs):
        m = super().from_config(*args, **kwargs)
        unet_add_concat_conds(unet=m, new_channels=4)
        unet_add_coded_conds(unet=m, added_number_count=1)
        return m

# Load models
model_name = 'lllyasviel/paints_undo_single_frame'
tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to(torch.float16)
vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(torch.bfloat16)
unet = ModifiedUNet.from_pretrained(model_name, subfolder="unet").to(torch.float16)

unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

video_pipe = LatentVideoDiffusionPipeline.from_pretrained(
    'lllyasviel/paints_undo_multi_frame',
    fp16=True
)

memory_management.unload_all_models([
    video_pipe.unet, video_pipe.vae, video_pipe.text_encoder, video_pipe.image_projection, video_pipe.image_encoder,
    unet, vae, text_encoder
])

k_sampler = KDiffusionSampler(
    unet=unet,
    timesteps=1000,
    linear_start=0.00085,
    linear_end=0.020,
    linear=True
)

def find_best_bucket(h, w, options):
    min_metric = float('inf')
    best_bucket = None
    for (bucket_h, bucket_w) in options:
        metric = abs(h * bucket_w - w * bucket_h)
        if metric <= min_metric:
            min_metric = metric
            best_bucket = (bucket_h, bucket_w)
    return best_bucket

@torch.inference_mode()
def encode_cropped_prompt_77tokens(txt: str):
    memory_management.load_models_to_gpu(text_encoder)
    cond_ids = tokenizer(txt,
                         padding="max_length",
                         max_length=tokenizer.model_max_length,
                         truncation=True,
                         return_tensors="pt").input_ids.to(device=text_encoder.device)
    text_cond = text_encoder(cond_ids, attention_mask=None).last_hidden_state
    return text_cond

@torch.inference_mode()
def pytorch2numpy(imgs):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)
        y = y * 127.5 + 127.5
        y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        results.append(y)
    return results

@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.5 - 1.0
    h = h.movedim(-1, 1)
    return h

def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)

@torch.inference_mode()
def interrogator_process(x):
    return wd14tagger.default_interrogator(x)

@torch.inference_mode()
def process(input_fg, prompt, input_undo_steps, image_width, image_height, seed, steps, n_prompt, cfg):
    rng = torch.Generator(device=memory_management.gpu).manual_seed(int(seed))

    memory_management.load_models_to_gpu(vae)
    fg = resize_and_center_crop(input_fg, image_width, image_height)
    
    # Ensure the input has 3 channels
    if fg.shape[2] == 4:
        fg = fg[:, :, :3]
    
    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

    memory_management.load_models_to_gpu(text_encoder)
    conds = encode_cropped_prompt_77tokens(prompt)
    unconds = encode_cropped_prompt_77tokens(n_prompt)

    memory_management.load_models_to_gpu(unet)
    fs = torch.tensor(input_undo_steps).to(device=unet.device, dtype=torch.long)
    initial_latents = torch.zeros_like(concat_conds)
    concat_conds = concat_conds.to(device=unet.device, dtype=unet.dtype)
    latents = k_sampler(
        initial_latent=initial_latents,
        strength=1.0,
        num_inference_steps=steps,
        guidance_scale=cfg,
        batch_size=len(input_undo_steps),
        generator=rng,
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        cross_attention_kwargs={'concat_conds': concat_conds, 'coded_conds': fs},
        same_noise_in_batch=True
    ).to(vae.dtype) / vae.config.scaling_factor

    memory_management.load_models_to_gpu(vae)
    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels)
    pixels = [fg] + pixels + [np.zeros_like(fg) + 255]

    return pixels

@torch.inference_mode()
def process_video_inner(image_1, image_2, prompt, seed=123, steps=25, cfg_scale=7.5, fs=3):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    frames = 16

    target_height, target_width = find_best_bucket(
        image_1.shape[0], image_1.shape[1],
        options=[(320, 512), (384, 448), (448, 384), (512, 320)]
    )

    image_1 = resize_and_center_crop(image_1, target_width=target_width, target_height=target_height)
    image_2 = resize_and_center_crop(image_2, target_width=target_width, target_height=target_height)
    input_frames = numpy2pytorch([image_1, image_2])
    input_frames = input_frames.unsqueeze(0).movedim(1, 2)

    memory_management.load_models_to_gpu(video_pipe.text_encoder)
    positive_text_cond = video_pipe.encode_cropped_prompt_77tokens(prompt)
    negative_text_cond = video_pipe.encode_cropped_prompt_77tokens("")

    memory_management.load_models_to_gpu([video_pipe.image_projection, video_pipe.image_encoder])
    input_frames = input_frames.to(device=video_pipe.image_encoder.device, dtype=video_pipe.image_encoder.dtype)
    positive_image_cond = video_pipe.encode_clip_vision(input_frames)
    positive_image_cond = video_pipe.image_projection(positive_image_cond)
    negative_image_cond = video_pipe.encode_clip_vision(torch.zeros_like(input_frames))
    negative_image_cond = video_pipe.image_projection(negative_image_cond)

    memory_management.load_models_to_gpu([video_pipe.vae])
    input_frames = input_frames.to(device=video_pipe.vae.device, dtype=video_pipe.vae.dtype)
    input_frame_latents, vae_hidden_states = video_pipe.encode_latents(input_frames, return_hidden_states=True)
    first_frame = input_frame_latents[:, :, 0]
    last_frame = input_frame_latents[:, :, 1]
    concat_cond = torch.stack([first_frame] + [torch.zeros_like(first_frame)] * (frames - 2) + [last_frame], dim=2)

    memory_management.load_models_to_gpu([video_pipe.unet])
    latents = video_pipe(
        batch_size=1,
        steps=int(steps),
        guidance_scale=cfg_scale,
        positive_text_cond=positive_text_cond,
        negative_text_cond=negative_text_cond,
        positive_image_cond=positive_image_cond,
        negative_image_cond=negative_image_cond,
        concat_cond=concat_cond,
        fs=fs
    )

    memory_management.load_models_to_gpu([video_pipe.vae])
    video = video_pipe.decode_latents(latents, vae_hidden_states)
    return video, image_1, image_2

@torch.inference_mode()
def process_video(keyframes, prompt, steps, cfg, fps, seed):
    result_frames = []
    cropped_images = []

    for i, (im1, im2) in enumerate(zip(keyframes[:-1], keyframes[1:])):
        im1 = np.array(Image.open(im1[0]))
        im2 = np.array(Image.open(im2[0]))
        frames, im1, im2 = process_video_inner(
            im1, im2, prompt, seed=seed + i, steps=steps, cfg_scale=cfg, fs=3
        )
        result_frames.append(frames[:, :, :-1, :, :])
        cropped_images.append([im1, im2])

    video = torch.cat(result_frames, dim=2)
    video = torch.flip(video, dims=[2])

    uuid_name = str(uuid.uuid4())
    output_filename = os.path.join(result_dir, uuid_name + '.mp4')
    Image.fromarray(cropped_images[0][0]).save(os.path.join(result_dir, uuid_name + '.png'))
    video = save_bcthw_as_mp4(video, output_filename, fps=fps)
    video = [x.cpu().numpy() for x in video]
    return output_filename, video

def main():
    parser = argparse.ArgumentParser(description="Generate key frames and video from an input image.")
    
    # Arguments for key frame generation
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--prompt', type=str, default="A beautiful painting", help='Prompt for key frame generation.')
    parser.add_argument('--input_undo_steps', type=int, nargs='+', default=[400, 600, 800, 900, 950, 999], help='Undo steps for key frame generation.')
    parser.add_argument('--image_width', type=int, default=512, help='Width of the generated images.')
    parser.add_argument('--image_height', type=int, default=640, help='Height of the generated images.')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed for key frame generation.')
    parser.add_argument('--steps', type=int, default=50, help='Number of steps for key frame generation.')
    parser.add_argument('--n_prompt', type=str, default='lowres, bad anatomy, bad hands, cropped, worst quality', help='Negative prompt for key frame generation.')
    parser.add_argument('--cfg', type=float, default=3.0, help='CFG scale for key frame generation.')

    # Arguments for video generation
    parser.add_argument('--video_prompt', type=str, default="1girl, masterpiece, best quality", help='Prompt for video generation.')
    parser.add_argument('--video_steps', type=int, default=50, help='Number of steps for video generation.')
    parser.add_argument('--video_cfg', type=float, default=7.5, help='CFG scale for video generation.')
    parser.add_argument('--video_fps', type=int, default=4, help='FPS for the generated video.')
    parser.add_argument('--video_seed', type=int, default=123, help='Random seed for video generation.')

    args = parser.parse_args()

    # Load input image
    input_fg = np.array(Image.open(args.image_path))
    
    # Generate key frames
    key_frames = process(
        input_fg, 
        args.prompt, 
        args.input_undo_steps, 
        args.image_width, 
        args.image_height, 
        args.seed, 
        args.steps, 
        args.n_prompt, 
        args.cfg
    )
    
    key_frame_paths = []
    for i, frame in enumerate(key_frames):
        key_frame_path = os.path.join(result_dir, f'key_frame_{i}.png')
        Image.fromarray(frame).save(key_frame_path)
        key_frame_paths.append((key_frame_path,))

    # Generate video
    video_path, video_frames = process_video(
        key_frame_paths, 
        args.video_prompt, 
        args.video_steps, 
        args.video_cfg, 
        args.video_fps, 
        args.video_seed
    )
    print(f'Video saved at: {video_path}')

if __name__ == "__main__":
    main()
