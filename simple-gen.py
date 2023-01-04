# make sure you're logged in with `huggingface-cli login`
import uuid
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("mps")

# Recommended if your computer has < 64 GB of RAM
pipe.enable_attention_slicing()

prompt = "a photo of a cute dog"

# First-time "warmup" pass (see explanation above)
_ = pipe(prompt, num_inference_steps=1)

# Results match those from the CPU device after the warmup pass.
images = pipe(prompt).images

for image in images:
    image_name = str(uuid.uuid1()) + ".jpg"
    image.save(image_name)

print("Done!")