import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

IMAGE_SIZE = 320
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.35),
        nn.Linear(model.classifier.in_features, 4)
    )
    return model.to(DEVICE)


def load_model(checkpoint_path):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path, map_location=DEVICE, weights_only=False
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = build_model()
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, checkpoint


def prep(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    x = transform(image).unsqueeze(0).to(DEVICE)
    resized_image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
    )
    return x, resized_image


def predict_grade(logits):
    ordinal_outputs = torch.sigmoid(logits[0])
    threshold_decisions = ordinal_outputs >= 0.5
    predicted_grade = int(threshold_decisions.sum().item())
    return predicted_grade, ordinal_outputs, threshold_decisions


def class_margin(logits, grade):
    if grade not in range(5):
        raise ValueError(f"Invalid KL grade: {grade}")
    if grade == 0:
        return -logits[:, 0]
    if grade == 4:
        return logits[:, 3]
    return logits[:, grade - 1] - logits[:, grade]


def gradcam(model, x, grade):
    activations, gradients = {}, {}
    target_layer = model.features.denseblock4

    def forward_hook(module, inputs, output):
        activations["value"] = output
        output.register_hook(
            lambda grad: gradients.__setitem__("value", grad)
        )

    hook_handle = target_layer.register_forward_hook(forward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        predicted_grade, _, _ = predict_grade(logits)
        score = class_margin(logits, grade).sum()
        score.backward()

        if "value" not in activations or "value" not in gradients:
            raise RuntimeError("Grad-CAM activations or gradients were not captured.")

        acts, grads = activations["value"], gradients["value"]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam, size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear", align_corners=False
        )
        cam = cam[0, 0].detach().cpu().numpy()
        cam_min, cam_max = float(cam.min()), float(cam.max())
        cam = cam - cam_min
        if cam_max - cam_min > 1e-8:
            cam = cam / (cam_max - cam_min)
        logits_np = logits.detach()[0].cpu().numpy()
        return cam, logits_np, predicted_grade
    finally:
        hook_handle.remove()


def colorize_heatmap(cam):
    color_stops = np.array([
        [0.0, 0.0, 80.0],
        [0.0, 160.0, 255.0],
        [0.0, 255.0, 180.0],
        [255.0, 230.0, 0.0],
        [255.0, 30.0, 0.0]
    ], dtype=np.float32)
    scaled = np.clip(cam, 0.0, 1.0) * (len(color_stops) - 1)
    index = np.floor(scaled).astype(int)
    index = np.clip(index, 0, len(color_stops) - 2)
    fraction = (scaled - index)[..., None]
    rgb = color_stops[index] * (1 - fraction) + color_stops[index + 1] * fraction
    return Image.fromarray(np.uint8(np.clip(rgb, 0, 255)), mode="RGB")


def build_analysis(grade, probabilities, decisions, cam):
    grade_labels = {
        0: ("No radiographic OA detected by the model", "The model's ordinal thresholds did not indicate a grade above 0."),
        1: ("Doubtful / very mild pattern", "One ordinal threshold was met: the model's output crossed the 50% cutoff for grade > 0, but not for higher thresholds."),
        2: ("Mild pattern", "Two ordinal thresholds were met: the model's outputs crossed the 50% cutoff for grade > 0 and grade > 1, but not for grade > 2."),
        3: ("Moderate pattern", "Three ordinal thresholds were met: the model's outputs crossed the 50% cutoff through grade > 2, but not grade > 3."),
        4: ("Severe pattern", "All four ordinal thresholds were met: each model output crossed the 50% cutoff, including grade > 3.")
    }
    title, threshold_reason = grade_labels[grade]
    threshold_rows = [
        {
            "threshold": f"P(grade > {k})",
            "probability": round(float(probabilities[k]) * 100, 2),
            "passed": bool(decisions[k]),
            "cutoff": 50
        }
        for k in range(4)
    ]
    peak = float(np.max(cam)) if cam.size else 0.0
    mean_activation = float(np.mean(cam)) if cam.size else 0.0
    return {
        "grade": grade,
        "label": title,
        "threshold_reason": threshold_reason,
        "method": (
            "The uploaded image was resized to 320 × 320, converted to a tensor, "
            "and normalized using ImageNet mean and standard deviation. A DenseNet-121 "
            "model produced four ordinal outputs. Each output was passed through a "
            "sigmoid; outputs at or above 50% were counted, and that count became the grade."
        ),
        "thresholds": threshold_rows,
        "gradcam": {
            "target_layer": "DenseNet-121 features.denseblock4",
            "description": (
                "Grad-CAM uses gradients of the selected grade's class-margin score "
                "to weight feature maps from the final dense block. The resulting map "
                "highlights image regions that influenced this model score; it does not "
                "identify a specific anatomical abnormality or prove a clinical finding."
            ),
            "mean_activation": round(mean_activation, 4),
            "max_activation": round(peak, 4)
        },
        "clinical_note": (
            "This is a model classification, not a confirmed radiology finding. "
            "The model output and Grad-CAM should be reviewed alongside the original "
            "X-ray and clinical context by a qualified healthcare professional."
        )
    }


def main():
    parser = argparse.ArgumentParser(description="MediPredict V5 Grad-CAM")
    parser.add_argument("image", help="Input knee X-ray image path")
    parser.add_argument("--checkpoint", default="models/best_model_v5.pth")
    parser.add_argument("--output", default="gradcam_v5")
    parser.add_argument("--alpha", type=float, default=0.42)
    args = parser.parse_args()

    if not 0 <= args.alpha <= 1:
        raise ValueError("--alpha must be between 0 and 1")

    print("\nLoading MediPredict V5 model...")
    model, checkpoint = load_model(args.checkpoint)
    print("Preprocessing X-ray...")
    x, original_image = prep(args.image)

    with torch.no_grad():
        logits = model(x)
        predicted_grade, ordinal_outputs, threshold_decisions = predict_grade(logits)

    print("Generating Grad-CAM...")
    cam, logits_np, grade_check = gradcam(model, x, predicted_grade)
    if grade_check != predicted_grade:
        raise RuntimeError("Prediction changed between inference and Grad-CAM.")

    heatmap = colorize_heatmap(cam)
    overlay = Image.blend(original_image, heatmap, alpha=args.alpha)
    heat_path = args.output + "_heatmap.png"
    overlay_path = args.output + "_overlay.png"
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    heatmap.save(heat_path)
    overlay.save(overlay_path)

    probabilities = ordinal_outputs.detach().cpu().numpy().tolist()
    decisions = threshold_decisions.detach().cpu().numpy().tolist()
    analysis = build_analysis(predicted_grade, probabilities, decisions, cam)

    print("\n" + "=" * 55)
    print("             MediPredict V5")
    print("       Knee Osteoarthritis Grading")
    print("=" * 55)
    print(f"Image             : {os.path.abspath(args.image)}")
    print(f"Checkpoint        : {args.checkpoint}")
    print(f"Device            : {DEVICE}")
    if isinstance(checkpoint, dict) and "epoch" in checkpoint:
        print(f"Checkpoint Epoch  : {checkpoint['epoch']}")
    print("-" * 55)
    print(f"Predicted KL Grade: {predicted_grade}")
    print("-" * 55)
    print("Ordinal Outputs P(grade > k):")
    for k, probability in enumerate(probabilities):
        decision = "YES" if decisions[k] else "NO"
        print(f"  P(grade > {k}): {probability * 100:.2f}%   | Threshold >= 50%: {decision}")
    print("-" * 55)
    print(f"Grad-CAM Heatmap : {heat_path}")
    print(f"Grad-CAM Overlay : {overlay_path}")
    print("=" * 55)
    print("Note: AI prediction is for research/demo purposes.")
    print("Grad-CAM is an exploratory visualization, not a clinical explanation.")
    print("=" * 55)

    # Machine-readable explanation for the Flask frontend; CMD output above stays intact.
    print("MEDIPREDICT_ANALYSIS_JSON=" + json.dumps(analysis, separators=(",", ":")))


if __name__ == "__main__":
    main()
