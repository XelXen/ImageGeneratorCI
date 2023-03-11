from diffusers import StableDiffusionPipeline
import torch
import requests
from json import loads

model_id = "dreamlike-art/dreamlike-photoreal-2.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

prompt = "photo, a church in the middle of a field of crops, bright cinematic lighting, gopro, fisheye lens"
image = pipe(prompt).images[0]

image.save("result.jpg")

status = loads(requests.get("https://api.gofile.io/getServer").text)
upload = loads(requests.post(f"https://{status['data']['server']}.gofile.io/uploadFile", files = {'file': open("result.jpg" ,'rb')}).text)['data']['downloadPage']

print(upload)
