from flask import Flask, request, send_file, jsonify
import io
import torch
from diffusers import StableDiffusionPipeline

app = Flask(__name__)

pipe = StableDiffusionPipeline.from_pretrained(
    "XpucT/Deliberate"
)

pipe.enable_attention_slicing()
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

@app.route('/generate', methods=['POST'])
def generate_image():
    prompt = request.json.get('prompt')
    if not prompt:
        return jsonify({'error': 'Prompt is required.'}), 400

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
    app.run(port=8080)