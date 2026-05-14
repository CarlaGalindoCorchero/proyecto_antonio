from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# Plantillas HTML
templates = Jinja2Templates(directory="templates")

# Cargar modelo .keras
model = tf.keras.models.load_model("models/modelo_final.keras")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    contents = await file.read()

    # Procesar imagen (ajusta según tu modelo)
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((224, 224))
    x = np.array(image) / 255.0
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x).tolist()

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "filename": file.filename,
            "prediction": pred
        }
    )
