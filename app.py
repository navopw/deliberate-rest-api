import torch
from diffusers import StableDiffusionPipeline
import openai
import uuid
import os
from dotenv import load_dotenv

RESULT_FOLDER = "result"

repo_id = "XpucT/Deliberate"
pipe = StableDiffusionPipeline.from_pretrained(repo_id)

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
Give me a stable diffusion prompt. Think of the theme by yourself.

Examples:
- strong warrior princess| centered| key visual| intricate| highly detailed| breathtaking beauty| precise lineart| vibrant| comprehensive cinematic| Carne Griffiths| Conrad Roset

- complex 3d render ultra detailed of a beautiful porcelain profile woman android face, cyborg, robotic parts, 150 mm, beautiful studio soft light, rim light, vibrant details, luxurious cyberpunk, lace, hyperrealistic, anatomical, facial muscles, cable electric wires, microchip, elegant, beautiful background, octane render, H. R. Giger style, 8k

- goddess close-up portrait skull with mohawk, ram skull, skeleton, thorax, x-ray, backbone, jellyfish phoenix head, nautilus, orchid, skull, betta fish, bioluminiscent creatures, intricate artwork by Tooth Wu and wlop and beeple. octane render, trending on artstation, greg rutkowski very coherent symmetrical artwork. cinematic, hyper realism, high detail, octane render, 8k

- aerial view of a giant fish tank shaped like a tower in the middle of new york city, 8k octane render, photorealistic

Maximum length 150 characters. Comma seperated keywords.
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
        num_inference_steps=22
    ).images[0]
    
    # Save image to path {RESULT_FOLDER}/{randomId}.png
    image_path = f"{RESULT_FOLDER}/{randomId}.png"
    image.save(image_path)

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