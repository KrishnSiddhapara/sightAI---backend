import os
import json
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types
from PIL import Image
import io

client = genai.Client(api_key=os.getenv('VLM_API_KEY'))

for img_name in ['scene1_tabletop.png', 'scene2_desk.png', 'scene3_figurines.png']:
    im = Image.open(f'tests/eval/images/{img_name}').convert('RGB')
    buf = io.BytesIO()
    im.save(buf, format='JPEG')
    part = types.Part.from_bytes(data=buf.getvalue(), mime_type='image/jpeg')
    
    prompt = (
        "Identify and localize EVERY distinct physical object in this image with an ultra-tight bounding box.\n"
        "Return a JSON list: [{\"label\": str, \"box_2d\": [ymin, xmin, ymax, xmax]}] on 0-1000 scale.\n"
        "Include all items: large and small (e.g. smartwatch, candles, mouse, figurines, plants, jars, laptops, books, cables)."
    )
    
    res = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[part, prompt],
        config=types.GenerateContentConfig(
            response_mime_type='application/json',
            temperature=0.0
        )
    )
    print(f"=== {img_name} ===")
    print(res.text)
