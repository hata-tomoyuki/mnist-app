from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from io import BytesIO
from fastapi import File, UploadFile
from PIL import Image

import numpy
import onnxruntime

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"))

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/api/predict")
async def predict(image: UploadFile = File(...)):
    image_data = await image.read()
    pil_image = Image.open(BytesIO(image_data))


    resized_image = pil_image.resize((28, 28))
    resized_arr = numpy.array(resized_image)
    transposed_arr = resized_arr.transpose(2, 0, 1)
    alpha_arr = transposed_arr[3]
    reshaped_arr = alpha_arr.reshape(-1)
    input = [reshaped_arr.astype(numpy.float32)]

    onnx_session = onnxruntime.InferenceSession("model.onnx")
    output = onnx_session.run(['probabilities'], {'float_input': input})
    result = output[0][0]

    return {'probabilities': result.tolist()}
