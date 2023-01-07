## Instructions

### Create file/directory for hugging-face token:

!mkdir -p ~/.huggingface
HUGGINGFACE_TOKEN = "YOUR_TOKEN"
!echo -n "{HUGGINGFACE_TOKEN}" > ~/.huggingface/token


### Install venv and deps:

python3 -m venv .
bin/pip3 install -r requirements.txt

### Add files to train AI:

Put 20 photos of you 512x512 pixels at instance_data_dir

bin/python3 train_dreambooth.py --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" --instance_data_dir="content/data/natan" --instance_prompt="natan" --class_data_dir="content/data/person" --class_prompt="photo of a person" --output_dir="content/stable_diffusion_weights/zwx" --revision="fp16" --with_prior_preservation --prior_loss_weight=1.0 --seed=1337 --resolution=512 --train_batch_size=1  --train_text_encoder --mixed_precision="no" --use_8bit_adam --gradient_accumulation_steps=1 --learning_rate=1e-6 --lr_scheduler="constant" --lr_warmup_steps=0 --num_class_images=50 --sample_batch_size=4 --max_train_steps=800

or 

accelerate launch train_dreambooth.py --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" --instance_data_dir="content/data/natan" --instance_prompt="natan" --class_data_dir="content/data/person" --class_prompt="photo of a person" --output_dir="content/stable_diffusion_weights/zwx" --revision="fp16" --with_prior_preservation --prior_loss_weight=1.0 --seed=1337 --resolution=512 --train_batch_size=1  --train_text_encoder --mixed_precision="no" --use_8bit_adam --gradient_accumulation_steps=1 --learning_rate=1e-6 --lr_scheduler="constant" --lr_warmup_steps=0 --num_class_images=50 --sample_batch_size=4 --max_train_steps=800

### Run:

bin/python3 <python_file>