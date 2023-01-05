# make sure you're logged in with `huggingface-cli login`
import uuid
import os
from diffusers import StableDiffusionPipeline

OUTPUT_DIR = "gen"

if not os.path.exists("%s" % OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None)
pipe = pipe.to("mps")

# Recommended if your computer has < 64 GB of RAM
pipe.enable_attention_slicing()

prompt = "a photo of a cute dog"

# First-time "warmup" pass (see explanation above)
_ = pipe(prompt, num_inference_steps=1)

# Results match those from the CPU device after the warmup pass.
for i in range(4):
    image = pipe(prompt).images[0]
    image_name = str(uuid.uuid1()) + ".jpg"
    image.save(OUTPUT_DIR + "/" + image_name)

print("Done!")