import io
import os
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from PIL import Image

from .pipeline import StainPipeline

# Environment-configurable defaults
# DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", None)
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", None)
# if DEFAULT_MODEL_NAME and DEFAULT_MODEL_PATH:
#     raise RuntimeError("Provide only one of MODEL_NAME or MODEL_PATH")

app = FastAPI(title="CycleGAN-Turbo Server", version="1.1")

# Lazy pipeline
_pipeline = None
def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = StainPipeline(
            # pretrained_name=DEFAULT_MODEL_NAME,
            pretrained_path=DEFAULT_MODEL_PATH,
        )
    return _pipeline

def _pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def _b64_png(img: Image.Image) -> str:
    return base64.b64encode(_pil_to_png_bytes(img)).decode("ascii")

@app.get("/health")
def health():
    return {"status": "ok"}

# JSON endpoint returning both input & output as base64
@app.post("/translate_json")
async def translate_json(
    image: UploadFile = File(...),
    model_path: str | None = Form(None),
):
    direction = "a2b"
    prompt = "patch from WSI of hematoxylin and eosin stain to be converted to patch of immunohistochemistry ER stain",
    seed = 42,
    
    try:
        raw = await image.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    pipe = get_pipeline()
    if model_path:
        pipe = StainPipeline(pretrained_path=model_path)

    try:
        out = pipe.run(img, direction=direction, prompt=prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse({
        "input":  {"mime": "image/png", "base64": _b64_png(img)},
        "output": {"mime": "image/png", "base64": _b64_png(out)},
        "meta": {
            "direction": direction,
            "seed": seed,
            "used_model": model_path or DEFAULT_MODEL_PATH,
        }
    })

# NEW: HTML preview showing both images side-by-side
@app.post("/translate_preview", response_class=HTMLResponse)
async def translate_preview(
    image: UploadFile = File(...),
    model_path: str | None = Form(None),
):
    direction = "a2b"
    prompt = "patch from WSI of hematoxylin and eosin stain to be converted to patch of immunohistochemistry ER stain",
    seed = 42,
    
    try:
        raw = await image.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    pipe = get_pipeline()
    if model_path:
        pipe = StainPipeline(pretrained_path=model_path)

    try:
        out = pipe.run(img, direction=direction, prompt=prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    input_data = _b64_png(img)
    output_data = _b64_png(out)
    used_model = model_path or DEFAULT_MODEL_PATH

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Translate Preview</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: start; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px; }}
    .title {{ font-weight: 600; margin-bottom: 8px; }}
    img {{ max-width: 100%; height: auto; border-radius: 8px; }}
    .meta {{ margin-top: 8px; color: #6b7280; font-size: 0.9rem; }}
    .back {{ margin-top: 16px; display: inline-block; }}
  </style>
</head>
<body>
  <h2>Result</h2>
  <div class="meta">model: <code>{used_model}</code> 
  <div class="grid" style="margin-top:16px">
    <div class="card">
      <div class="title">H&E patch</div>
      <img src="data:image/png;base64,{input_data}" alt="input" />
    </div>
    <div class="card">
      <div class="title">ER patch</div>
      <img src="data:image/png;base64,{output_data}" alt="output" />
    </div>
  </div>
  <a class="back" href="/">&#8592; back</a>
</body>
</html>
"""
    return HTMLResponse(html)

# NEW: simple index with an upload form posting to /translate_preview
@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse("""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>H&E to ER IHC translation</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }
    form { display: grid; gap: 10px; max-width: 520px; }
    label { font-weight: 600; }
    input, button { padding: 8px; }
    .hint { color: #6b7280; font-size: 0.9rem; }
  </style>
</head>
<body>
  <h2>Upload an H&E iamge patch</h2>
  <form action="/translate_preview" method="post" enctype="multipart/form-data">
    <label>Image <input type="file" name="image" required /></label>
    <button type="submit">Run</button>
  </form>
</body>
</html>
""")
