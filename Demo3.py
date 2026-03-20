import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import gradio as gr
import numpy as np

# -----------------------------
# MODEL SETUP (ImageNet)
# -----------------------------
MODEL_ID = "google/vit-base-patch16-224"

processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
model.eval()
labels = model.config.id2label


# -----------------------------
# Prediction
# -----------------------------
def classify_image(img):
    if img is None:
        return {"No image": 1.0}

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype("uint8")).convert("RGB")
    else:
        img = img.convert("RGB")

    inputs = processor(images=img, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits[0], dim=0)

    topk = torch.topk(probs, 5)
    return {labels[int(i)]: float(v) for v, i in zip(topk.values, topk.indices)}


# -----------------------------
# Professional UI
# -----------------------------
with gr.Blocks(title="AI Photo Classifier", css="""
    .title {text-align: center; font-size: 28px; font-weight: bold; margin-bottom: 8px;}
    .subtitle {text-align: center; color: #777; margin-bottom: 25px;}
""") as demo:

    gr.HTML("<div class='title'>AI Photo Classifier</div>")
    gr.HTML("<div class='subtitle'>Upload a photo to get high-accuracy ImageNet predictions</div>")

    with gr.Row():
        with gr.Column(scale=1):
            uploader = gr.Image(type="pil", label="Upload Photo", height=350)
            classify_btn = gr.Button("Classify Image", variant="primary")

        with gr.Column(scale=1):
            results = gr.Label(num_top_classes=5, label="Predictions (Confidence)")

    classify_btn.click(fn=classify_image, inputs=uploader, outputs=results)

demo.launch()
