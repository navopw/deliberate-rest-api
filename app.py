import torch
from diffusers import StableDiffusionPipeline, StableDiffusionUpscalePipeline
import openai
import uuid
import os
from dotenv import load_dotenv

RESULT_FOLDER = "result"

# Deliberate Pipeline
repo_id = "XpucT/Deliberate"
pipe = StableDiffusionPipeline.from_pretrained(repo_id)

# Upscale pipeline
upscale_pipe = StableDiffusionUpscalePipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler")

def get_best_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def send_message(message_log):
    # Use OpenAI's ChatCompletion API to get the chatbot's response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # The name of the OpenAI chatbot model to use
        messages=message_log,   # The conversation history up to this point, as a list of dictionaries
        max_tokens=3000,        # The maximum number of tokens (words or subwords) in the generated response
        stop=None,              # The stopping sequence for the generated response, if any (not used here)
        temperature=0.7,        # The "creativity" of the generated response (higher temperature = more creative)
    )

    # Find the first response from the chatbot that has text in it (some responses may not have text)
    for choice in response.choices:
        if "text" in choice:
            return choice.text

    # If no response with text is found, return the first response's content (which may be empty)
    return response.choices[0].message.content

def generate_caption(prompt):
    # Generate caption
    completion = send_message([
        {"role": "user","content": """
Act as a Instagram caption generator, I give you the prompt of the image and you output a short caption and the fitting hashtags to reach as many interested people as possible. Try to make it a bit alternative.
Answer with ONLY the caption. PROMPT: """ + prompt},
    ])

    # Trim backslash n and whitespaces
    completion = completion.replace("\n", "").strip()
    completion = completion.replace('"', "").strip()

    return completion

def generate_prompt():
    # Generate prompt
    completion = send_message([
        {"role": "user", "content": """
Stable Diffusion is an AI art generation model similar to DALLE-2.
Below is a list of prompts that can be used to generate images with Stable Diffusion:

- a cute kitten made out of metal, (cyborg:1.1), ([tail | detailed wire]:1.3), (intricate details), hdr, (intricate details, hyperdetailed:1.2), cinematic shot, vignette, centered
- medical mask, victorian era, cinematography, intricately detailed, crafted, meticulous, magnificent, maximum details, extremely hyper aesthetic
- a girl, wearing a tie, cupcake in her hands, school, indoors, (soothing tones:1.25), (hdr:1.25), (artstation:1.2), dramatic, (intricate details:1.14), (hyperrealistic 3d render:1.16), (filmic:0.55), (rutkowski:1.1), (faded:1.3)
- A ultra detailed portrait of a sailor moon 1girl smiling, color digital painting, highly detailed, digital painting, artstation, intricate, sharp focus, warm lighting, attractive, high quality, masterpiece, award-winning art, art by Yoshitaka Amano, and Brian Froud, trending on artstation, trending on deviantart, Anime Key Visual, anime coloring, (anime screencap:1.2),(Graphic Novel),(style of anime:1.3), trending on CGSociety
- (extremely detailed CG unity 8k wallpaper), movie poster of elite (Beskar Steel pattern:1.6) intricate armor mandalorian in a battle stance, professional majestic oil painting by Ed Blinkey, Atey Ghailan, Studio Ghibli, by Jeremy Mann, Greg Manchess, Antonio Moro, trending on ArtStation, trending on CGSociety, Intricate, High Detail, Sharp focus, dramatic, by midjourney and greg rutkowski, realism, beautiful and detailed lighting, shadows, by Jeremy Lipking, by Antonio J. Manzanedo, by Frederic Remington, by HW Hansen, by Charles Marion Russell, by William Herbert Dunton
- Giant monstrous aliens, volumetric lighting, concept art, smooth, sharp focus, 8k octane beautifully detailed render, post-processing, extremely hyperdetailed, intricate, epic composition, grim yet sparkling atmosphere, cinematic lighting + masterpiece, trending on artstation, very detailed, vibrant colors
- masterpiece,best quality,official art,extremely detailed CG unity 8k wallpaper,illustration, light,car, bright, motor vehicle, ground vehicle, sports car, vehicle focus, road, need for speed, moving, wet, (night, midnight:1.5), cyberpunk, tokyo,neon lights,drift, <lora:MX5NA-000008:0.9>
- cyborg woman| with a visible detailed brain| muscles cable wires| biopunk| cybernetic| cyberpunk| white marble bust| canon m50| 100mm| sharp focus| smooth| hyperrealism| highly detailed| intricate details| carved by michelangelo

I want you to write me a a detailed prompt. Follow the structure of the example prompts. This means a very short description of the scene as tags, followed by modifiers divided by commas to alter the mood, style, lighting, and more. Also append hdr, 8k, high detail, high quality. Dont surround with parenthesis.
         """},
    ])
    
    # Trim backslash n and whitespaces
    completion = completion.replace("\n", "").strip()
    completion = completion.replace('"', "").strip()
    
    return completion

def generate():
    # UUID
    randomId = uuid.uuid4()
    
    # Generate prompt
    prompt = generate_prompt()
    print(f"[Prompt] {prompt}")

    # Generate caption
    caption = generate_caption(prompt)
    print(f"[Caption] {caption}")

    # Save prompt, caption as json
    with open(f"{RESULT_FOLDER}/{randomId}.json", "w", encoding="utf-8") as f:
        f.write(f'{{"prompt": "{prompt}", "caption": "{caption}"}}')

    # Generate Image
    image = pipe(
        prompt=prompt,
        width=512,
        height=512,
        num_inference_steps=30
    ).images[0]
    
    # Upscale
    upscaled_image = upscale_pipe(
       prompt=prompt, image=image
    ).images[0]
    
    # Save image to path {RESULT_FOLDER}/{randomId}.png
    image_path = f"{RESULT_FOLDER}/{randomId}.png"
    upscaled_image.save(image_path)

if __name__ == '__main__':
    # Env
    load_dotenv()
    
    # OpenAI
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    
    print(f"[Torch] Using device: {get_best_device()}")

    # Diffusion
    pipe.to(get_best_device())
    pipe.enable_attention_slicing()

    # Create result folder if not exists
    if not os.path.exists(RESULT_FOLDER):
        os.makedirs(RESULT_FOLDER)

    # Run generation
    while True:
        generate()
