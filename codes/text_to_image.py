# requirements openai
# pip install openai

import base64
from openai import OpenAI
from super_secrets import OPENAI_API_KEY


client = OpenAI(api_key=OPENAI_API_KEY)

imagen = ""

response = client.images.edit(
    model="gpt-image-1",
    image=open(imagen, "rb"),
    prompt="The Simpsons style"
)

image_base64 = response.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

# Save the image to a file
with open("doomentio_los_sinsons.png", "wb") as f:
    f.write(image_bytes)