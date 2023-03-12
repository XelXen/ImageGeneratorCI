from diffusers import StableDiffusionPipeline
import torch
import requests
from json import loads

model_id = "dreamlike-art/dreamlike-photoreal-2.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = pipe.to(device)

with open('prompt.txt', 'r') as file:
    prompt = file.read()
    print(f"Prompt: {prompt.strip()}")

image = pipe(prompt).images[0]
image.save("result.jpg")

status = loads(requests.get("https://api.gofile.io/getServer").text)
upload = loads(requests.post(f"https://{status['data']['server']}.gofile.io/uploadFile", files = {'file': open("result.jpg" ,'rb')}).text)['data']['downloadPage']

print(f"\nDownload Link: {upload}")
