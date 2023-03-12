from flask import Flask, request, send_file, jsonify
import io
import torch
from diffusers import StableDiffusionPipeline
import openai
import uuid

app = Flask(__name__)

repo_id = "XpucT/Deliberate"
pipe = StableDiffusionPipeline.from_pretrained(repo_id)

def get_best_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"


def send_message(message_log):
    # Use OpenAI's ChatCompletion API to get the chatbot's response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # The name of the OpenAI chatbot model to use
        messages=message_log,   # The conversation history up to this point, as a list of dictionaries
        max_tokens=3800,        # The maximum number of tokens (words or subwords) in the generated response
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
        {"role": "user", "content": "Act as a Instagram caption generator, I give you the prompt of the image and you output a short caption and the fitting hashtags to reach as many interested people as possible. Try to make it a bit alternative. Answer with ONLY the caption. PROMPT: " + prompt},
    ])

    # Trim backslash n and whitespaces
    completion = completion.replace("\n", "").strip()
    completion = completion.replace('"', "").strip()

    return completion

@app.route('/generate', methods=['POST'])
def generate_image():
    prompt = request.json.get('prompt')
    if not prompt:
        return jsonify({'error': 'Prompt is required.'}), 400

    # UUID
    randomId = uuid.uuid4()

    # Generate caption
    caption = generate_caption(prompt)
    print(f"[Caption] {caption}")

    # Save prompt, caption as json
    with open(f"result/{randomId}.json", "w") as f:
        f.write(f'{{"prompt": "{prompt}", "caption": "{caption}"}}')

    # Image
    image = pipe(
        prompt=prompt,
        width=768,
        height=768,
        num_inference_steps=25
    ).images[0]
    
    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    # OpenAI
    openai.api_key = "sk-"

    # Diffusion
    pipe.to(get_best_device())
    pipe.enable_attention_slicing()

    app.run(port=8080)