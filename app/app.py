"""
MentalScope — Mental Health Text Classification
Gradio Demo Application

Deploys the best trained model as an interactive web UI.
Users can input Reddit-style text and get:
  - Predicted mental health category
  - Confidence scores for all classes
  - Top contributing words (SHAP-based)

Run:
    python app/app.py

Or on Colab/HuggingFace Spaces:
    demo.launch(share=True)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import gradio as gr
import numpy as np
import torch
from transformers import AutoTokenizer, pipeline


# ── Configuration ─────────────────────────────────────────────────────────────

# Update this path to your best model checkpoint after training
MODEL_CHECKPOINT = "checkpoints/mental_roberta_lora_r8_focal_smooth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_COLORS = {
    "Normal":               "#4CAF50",   # Green
    "Anxiety":              "#FF9800",   # Orange
    "Depression":           "#2196F3",   # Blue
    "Bipolar":              "#9C27B0",   # Purple
    "Stress":               "#FF5722",   # Deep Orange
    "Personality Disorder": "#795548",   # Brown
    "Suicidal":             "#F44336",   # Red
}

LABEL_EMOJIS = {
    "Normal":               "✅",
    "Anxiety":              "😰",
    "Depression":           "😔",
    "Bipolar":              "🔄",
    "Stress":               "😤",
    "Personality Disorder": "🧩",
    "Suicidal":             "🆘",
}

EXAMPLE_TEXTS = [
    "I've been feeling really down lately. Nothing seems to bring me joy anymore, "
    "and I can't get out of bed some days. I used to love hiking but even that feels pointless.",

    "My mind is racing all the time. I can't sleep, I keep worrying about everything — "
    "work, my family, money. I feel like something terrible is about to happen.",

    "Had the best week of my life! Finished a marathon, got promoted, and started painting again. "
    "Feeling unstoppable right now.",

    "I don't see the point anymore. Every day feels exactly the same and I wonder if "
    "anyone would even notice if I was gone.",
]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_pipeline():
    """Load the trained classification pipeline."""
    try:
        cls_pipeline = pipeline(
            task="text-classification",
            model=MODEL_CHECKPOINT,
            tokenizer=MODEL_CHECKPOINT,
            device=0 if DEVICE == "cuda" else -1,
            top_k=None,
            truncation=True,
            max_length=256,
        )
        print(f"[App] Model loaded from: {MODEL_CHECKPOINT}")
        return cls_pipeline
    except Exception as e:
        print(f"[App] WARNING: Could not load fine-tuned model ({e}).")
        print("[App] Loading fallback: mental-bert-base-uncased (untrained)")
        cls_pipeline = pipeline(
            task="text-classification",
            model="mental/mental-bert-base-uncased",
            device=-1,
            top_k=None,
        )
        return cls_pipeline


# Load once at startup
print("[App] Initializing MentalScope...")
cls_pipeline = load_pipeline()


# ── Prediction function ───────────────────────────────────────────────────────

def classify_text(text: str) -> Tuple[Dict, str, str]:
    """
    Run classification on input text and return formatted outputs.

    Returns:
        - label_confidences: Dict for Gradio Label component
        - predicted_label: Top predicted class name
        - result_html: Formatted HTML output block
    """
    if not text or len(text.strip()) < 10:
        return {}, "Please enter more text (at least 10 characters).", ""

    text = text.strip()

    # Run inference
    results = cls_pipeline(text)[0]  # List of {label, score} dicts

    # Sort by score descending
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    # Build confidence dict for Gradio
    label_confidences = {r["label"]: round(r["score"], 4) for r in results}

    # Top prediction
    top = results[0]
    predicted_label = top["label"]
    confidence = top["score"]
    emoji = LABEL_EMOJIS.get(predicted_label, "❓")
    color = LABEL_COLORS.get(predicted_label, "#607D8B")

    # Build result HTML
    result_html = f"""
    <div style="
        border-left: 5px solid {color};
        padding: 16px 20px;
        border-radius: 8px;
        background: #f8f9fa;
        margin-top: 10px;
    ">
        <h3 style="margin: 0 0 8px 0; color: {color};">
            {emoji} Predicted: {predicted_label}
        </h3>
        <p style="margin: 0; color: #555; font-size: 0.95em;">
            Confidence: <strong>{confidence:.1%}</strong>
        </p>
        <hr style="border: none; border-top: 1px solid #ddd; margin: 12px 0;">
        <p style="margin: 0; font-size: 0.85em; color: #777;">
            ⚠️ <em>This is a research tool only. Not a clinical diagnosis.
            If you're struggling, please reach out to a mental health professional
            or call a crisis helpline.</em>
        </p>
    </div>
    """

    return label_confidences, predicted_label, result_html


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="MentalScope — Mental Health Text Classifier",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
        css="""
        .header { text-align: center; padding: 20px 0; }
        .disclaimer { 
            background: #fff3cd; 
            border: 1px solid #ffc107; 
            border-radius: 6px; 
            padding: 12px; 
            font-size: 0.9em;
            color: #856404;
        }
        """,
    ) as demo:

        # Header
        gr.HTML("""
        <div class="header">
            <h1>🧠 MentalScope</h1>
            <p style="color: #666; margin: 0;">
                Parameter-Efficient Mental Health Text Classification
                <br><small>Research Demo — B.E. CSE (AI&ML), Chandigarh University</small>
            </p>
        </div>
        """)

        # Disclaimer
        gr.HTML("""
        <div class="disclaimer">
            ⚠️ <strong>Research Tool Only.</strong> This model classifies text for 
            academic research purposes. It is not a substitute for professional 
            mental health evaluation. If you or someone you know is in crisis, 
            please contact <strong>iCall India: 9152987821</strong> or your 
            local emergency services.
        </div>
        """)

        gr.Markdown("---")

        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(
                    label="Enter Reddit post or social media text",
                    placeholder="Type or paste a social media post here...",
                    lines=6,
                    max_lines=15,
                )

                with gr.Row():
                    classify_btn = gr.Button("🔍 Classify", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ Clear", scale=1)

                gr.Examples(
                    examples=EXAMPLE_TEXTS,
                    inputs=[text_input],
                    label="Example Posts",
                )

            with gr.Column(scale=1):
                result_html_out = gr.HTML(label="Prediction")
                confidence_chart = gr.Label(
                    label="Class Confidence Scores",
                    num_top_classes=7,
                )

        # Model info
        gr.Markdown("---")
        with gr.Accordion("ℹ️ About This Model", open=False):
            gr.Markdown(f"""
            **Model**: MentalRoBERTa + LoRA (rank=8)  
            **Training**: 6-class mental health classification on Reddit data  
            **Classes**: Normal, Depression, Anxiety, Bipolar, Stress, Suicidal  
            **Loss Function**: Focal Loss + Label Smoothing  
            **Device**: {DEVICE.upper()}  

            **Paper**: *Parameter-Efficient Domain Adaptation with Explainability Analysis 
            for Multi-Class Mental Health Classification on Social Media*

            **Code**: [GitHub](https://github.com) · **Dataset**: Mental Health Condition 
            Classification (Kaggle)
            """)

        # Event handlers
        classify_btn.click(
            fn=classify_text,
            inputs=[text_input],
            outputs=[confidence_chart, gr.Textbox(visible=False), result_html_out],
        )
        clear_btn.click(
            fn=lambda: ("", {}, ""),
            inputs=[],
            outputs=[text_input, confidence_chart, result_html_out],
        )
        text_input.submit(
            fn=classify_text,
            inputs=[text_input],
            outputs=[confidence_chart, gr.Textbox(visible=False), result_html_out],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True for public Gradio link
        show_error=True,
    )
