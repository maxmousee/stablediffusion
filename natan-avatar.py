# make sure you're logged in with `huggingface-cli login`
import uuid
import os
import json
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
from IPython.display import display


HUGGING_FACE = "~/.huggingface"

OUTPUT_DIR = "content/stable_diffusion_weights/zwx"

MODEL_NAME = "runwayml/stable-diffusion-v1-5"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[*] Weights will be saved at {OUTPUT_DIR}")

os.makedirs(MODEL_NAME, exist_ok=True)

concepts_list = [
    {
        "instance_prompt":      "natan",
        "class_prompt":         "photo of a person",
        "instance_data_dir":    "content/data/natan",
        "class_data_dir":       "content/data/person"
    },
]

for c in concepts_list:
    os.makedirs(c["instance_data_dir"], exist_ok=True)

with open("concepts_list.json", "w") as f:
    json.dump(concepts_list, f, indent=4)

WEIGHTS_DIR = "stable_diffusion_weights/zwx"
if WEIGHTS_DIR == "":
    from natsort import natsorted
    from glob import glob
    import os
    WEIGHTS_DIR = natsorted(glob(OUTPUT_DIR + os.sep + "*"))[-1]
print(f"[*] WEIGHTS_DIR={WEIGHTS_DIR}")

weights_folder = OUTPUT_DIR
folders = sorted([f for f in os.listdir(weights_folder) if f != "0"], key=lambda x: int(x))

row = len(folders)
col = len(os.listdir(os.path.join(weights_folder, folders[0], "samples")))
scale = 4
fig, axes = plt.subplots(row, col, figsize=(col * scale, row * scale), gridspec_kw={'hspace': 0, 'wspace': 0})

# Run to generate a grid of preview images from the last saved weights.
for i, folder in enumerate(folders):
    folder_path = os.path.join(weights_folder, folder)
    image_folder = os.path.join(folder_path, "samples")
    images = [f for f in os.listdir(image_folder)]
    for j, image in enumerate(images):
        if row == 1:
            currAxes = axes[j]
        else:
            currAxes = axes[i, j]
        if i == 0:
            currAxes.set_title(f"Image {j}")
        if j == 0:
            currAxes.text(-0.1, 0.5, folder, rotation=0, va='center', ha='center', transform=currAxes.transAxes)
        image_path = os.path.join(image_folder, image)
        img = mpimg.imread(image_path)
        currAxes.imshow(img, cmap='gray')
        currAxes.axis('off')

plt.tight_layout()
plt.savefig('grid.png', dpi=72)

# Inference

model_path = "stable_diffusion_weights/zwx/800"

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
pipe = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler, safety_checker=None, torch_dtype=torch.float16).to("cuda")

prompt = " Beautiful warmly lit close up studio portrait of natan person sweetly smiling cute, impasto oil painting heavy brushstrokes by Cy Twombly and Anselm Kiefer , trending on artstation dramatic lighting abstract Expressionism"
negative_prompt = ""
num_samples = 6
guidance_scale = 5
num_inference_steps = 50
height = 512
width = 512

with autocast("cuda"), torch.inference_mode():
    images = pipe(
        prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_samples,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=None
    ).images

for img in images:
    display(img)
    image_name = str(uuid.uuid1()) + ".jpg"
    img.save(OUTPUT_DIR + "/" + image_name)

print("Done!")