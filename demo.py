# ============================================================
#  Age Classification - Live Demo Script
#  Run this locally in VS Code after downloading trained model
#  Austin Steffes - dotCon Demo
#
#  Updated to match the trained model:
#    - Model has 6 classes, not 12. Keras assigned them in
#      ALPHABETICAL order during training, so CLASS_NAMES must
#      be exactly:
#        ['child', 'infant', 'middle_age', 'senior', 'teen', 'young_adult']
#      (this is printed in every training log under "Classes (6):").
#      The order is critical: model output index 0 = child,
#      index 1 = infant, etc. Get this wrong and every label is
#      silently shuffled.
#    - EfficientNetV2 has preprocessing built into the model, so
#      we feed RAW 0-255 pixels. Do NOT divide by 255 here.
#    - Collapsed the two duplicate prediction functions into one.
# ============================================================

import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# -- CONFIG --------------------------------------------------
# Update this to wherever you downloaded the model from Drive
MODEL_PATH = r"C:\Users\astef\OneDrive\Desktop\Projects\dotCon\model_age_demo.keras"
IMAGE_SIZE = (224, 224)

# Must match the order Keras used during training (alphabetical).
# Confirmed from the training log: "Classes (6): [...]"
CLASS_NAMES = ['child', 'infant', 'middle_age', 'senior', 'teen', 'young_adult']

# Human-readable labels for display
AGE_LABELS = {
    'child':       'Child  (~4-12)',
    'infant':      'Infant  (~0-3)',
    'middle_age':  'Middle Age  (~36-55)',
    'senior':      'Senior  (~56+)',
    'teen':        'Teen  (~13-19)',
    'young_adult': 'Young Adult  (~20-35)',
}

# Display order for the chart: chronological by age, which reads
# more naturally than the alphabetical model order. We map back to
# model indices when pulling probabilities, so this is purely cosmetic.
DISPLAY_ORDER = ['infant', 'child', 'teen', 'young_adult', 'middle_age', 'senior']

# -- LOAD MODEL ----------------------------------------------
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

# Sanity check: model output size must match CLASS_NAMES length.
n_out = model.output_shape[-1]
if n_out != len(CLASS_NAMES):
    raise ValueError(
        f"Model outputs {n_out} classes but CLASS_NAMES has "
        f"{len(CLASS_NAMES)}. These must match. Check the training "
        f"log's 'Classes (N):' line and fix CLASS_NAMES."
    )
print(f"Model expects {n_out} classes. CLASS_NAMES matches. Good to go.\n")


# -- HELPERS -------------------------------------------------
def _save_temp(pil_image):
    """Save PIL image to a temp file for load_img compatibility."""
    path = "demo_input.jpg"   # local working dir; portable across OSes
    pil_image.save(path)
    return path


# -- PREDICTION FUNCTION -------------------------------------
def predict_age(image):
    """
    Takes a PIL image from Gradio, runs it through the model,
    returns (confidence bar chart, results summary string).
    """
    if image is None:
        return None, "No image provided."

    # Preprocess: resize to model input size, keep RAW 0-255 pixels.
    # EfficientNetV2 rescales internally, so we do NOT divide by 255.
    temp_path = _save_temp(image)
    img_arr = np.array(
        load_img(temp_path, target_size=IMAGE_SIZE)
    ).reshape(1, *IMAGE_SIZE, 3).astype("float32")

    # Predict -> 6 probabilities, one per class in CLASS_NAMES order
    probs = model.predict(img_arr, verbose=0)[0]

    # Index of the most confident class
    top_idx   = int(np.argmax(probs))
    top_class = CLASS_NAMES[top_idx]
    top_prob  = probs[top_idx] * 100
    top_label = AGE_LABELS[top_class]

    # -- Build results text (sorted high -> low confidence) --
    sorted_idx = np.argsort(probs)[::-1]
    result_text  = f"Top Prediction: {top_label}  ({top_prob:.1f}% confidence)\n\n"
    result_text += "Full Breakdown:\n"
    result_text += "-" * 40 + "\n"
    for idx in sorted_idx:
        label = AGE_LABELS[CLASS_NAMES[idx]]
        prob  = probs[idx] * 100
        bar   = "#" * int(prob / 5)
        result_text += f"{label:<22} {prob:5.1f}%  {bar}\n"

    # -- Build confidence bar chart (chronological display order) --
    display_labels = [AGE_LABELS[c] for c in DISPLAY_ORDER]
    display_probs  = [probs[CLASS_NAMES.index(c)] * 100 for c in DISPLAY_ORDER]
    colors = ["#2ecc71" if c == top_class else "#3498db" for c in DISPLAY_ORDER]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(display_labels, display_probs, color=colors,
                   edgecolor="white", height=0.6)
    ax.invert_yaxis()   # youngest at top

    for bar, prob in zip(bars, display_probs):
        if prob > 2:
            ax.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{prob:.1f}%",
                va="center", ha="left", fontsize=9, color="white"
            )

    ax.set_xlim(0, 110)
    ax.set_xlabel("Confidence (%)", color="white", fontsize=11)
    ax.set_title(f"Predicted: {top_label}  ({top_prob:.1f}%)",
                 color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#555")
    ax.spines["left"].set_color("#555")
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")
    plt.tight_layout()

    return fig, result_text


# -- GRADIO UI -----------------------------------------------
demo = gr.Interface(
    fn=predict_age,
    inputs=gr.Image(
        type="pil",
        label="Drop a photo here or click to upload",
    ),
    outputs=[
        gr.Plot(label="Confidence Chart"),
        gr.Textbox(label="Results", lines=12, max_lines=16),
    ],
    title="Age Classifier - dotCon Demo",
    description=(
        "Upload any face photo and the model predicts which age group it "
        "belongs to.\n"
        "Classes: Infant (0-3), Child (4-12), Teen (13-19), "
        "Young Adult (20-35), Middle Age (36-55), Senior (56+).\n"
        "Note: the model is most confident on infants and seniors, and "
        "naturally less certain between adjacent adult groups - that's "
        "expected, since those ages look similar even to people."
    ),
    theme=gr.themes.Soft(),
    flagging_mode="never",
)

# -- LAUNCH --------------------------------------------------
if __name__ == "__main__":
    print("\nStarting demo...")
    print("Once running, open the local URL in your browser.")
    print("Anyone on the same network can use the share link.\n")
    demo.launch(
        share=True,   # generates a public link for audience members
        show_error=True,
    )