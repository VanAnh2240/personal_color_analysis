from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile, shutil, cv2, numpy as np, sys
from pathlib import Path
import base64
import io
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from classification import PaletteClassifier
from classification.visualizer import save_result_figure

app = FastAPI()

DEFAULT_CHECKPOINT = "checkpoints/system_1_deeplabv3.pth"
_model = None


def get_model():
    global _model
    if _model is None:
        import torch
        import torch.nn.functional as F
        from src.models.system_1_deeplabv3 import DeepLabV3

        ckpt = torch.load(DEFAULT_CHECKPOINT, map_location="cpu")
        m = DeepLabV3(num_classes=11)
        m.load_state_dict(ckpt.get("model", ckpt))
        m.eval()
        _model = m

    return _model


@app.get("/health")
def health():
    return {"status": "ok", "service": "personal_color"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    import torch
    import torch.nn.functional as F

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    bgr = cv2.imread(tmp_path)
    if bgr is None:
        return JSONResponse({"error": "Cannot read image"}, status_code=400)

    # ===== SEGMENTATION =====
    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    _STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    orig_h, orig_w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (473, 473))

    tensor = (
        torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        - _MEAN
    ) / _STD

    with torch.no_grad():
        out = get_model()(tensor.unsqueeze(0))
        logits = out["out"] if isinstance(out, dict) else out
        logits = F.interpolate(
            logits,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )
        seg_mask = logits.argmax(1).squeeze(0).numpy().astype(np.uint8)

    # ===== CLASSIFICATION =====
    clf = PaletteClassifier(hair_label=10)
    result = clf.classify(bgr, seg_mask)
    
    img = save_result_figure(bgr, result, return_image=True)

    buffer = io.BytesIO()
    Image.fromarray(img).save(buffer, format="PNG")
    img_bytes = buffer.getvalue()

    return {
        "season": result.season.name,
        "metrics": result.metrics,
        "user_vector": result.user_vector,
        "is_bald": result.is_bald,
        "dominants": {k: list(map(int, v)) for k, v in result.dominants.items()},
        "result_image_base64": base64.b64encode(img_bytes).decode()
    }