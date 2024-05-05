from flask import Flask, request, render_template, jsonify
from diffusers import StableDiffusionPipeline
from transformers import pipeline, set_seed
import torch
import cv2
import os

app = Flask(__name__)

class CFG:
    seed = 42
    if torch.cuda.is_available():
        device = "cuda"
        generator = torch.Generator(device).manual_seed(seed)
    else:
        device = "cpu"
        generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 100
    image_gen_model_id = "runwayml/stable-diffusion-v1-5"
    image_gen_size = (500,500)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt3"
    prompt_dataset_size = 6
    prompt_max_length = 12

def get_model():
    image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float16,
    variant="fp16", use_auth_token='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', guidance_scale=9,safety_checker = None
)
    image_gen_model = image_gen_model.to(CFG.device)
    return image_gen_model

def generate_image(prompt, model):
    image = model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]

    image = image.resize(CFG.image_gen_size)
    return image

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form.get('prompt')
    model = get_model()
    img = generate_image(prompt, model)
    save_path = save_image(img)

    response = {
        'image_path': save_path
    }

    return jsonify(response)

def save_image(image):
    save_folder = 'static'

    os.makedirs(save_folder, exist_ok=True)

    image_path = os.path.join(save_folder,'image.jpg')
    image.save(image_path)

    return image_path



@app.route('/')
def index():
    return render_template('index.html')



if __name__=='__main__':
    app.run(debug=True)