"""Main Gradio UI entrypoint for extraction, correction, and analytics."""

from __future__ import annotations

import argparse
import base64
from html import escape as html_escape
import json
import os
import re
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.active_learning import append_cycle_record, build_cycle_record
from src.correction_io import append_jsonl
from src.correction_schema import normalize_payload
from src.correction_state import CorrectionState
from src.metrics import extract_json
from src.model import get_device
from src.preprocess import build_prompt

# Avoid noisy tokenizer fork warnings in Gradio worker threads/processes.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Avoid Gradio SSR/event-loop issues in some Spaces Python 3.13 runtimes.
os.environ.setdefault("GRADIO_SSR_MODE", "false")
# Disable hot-reload hooks that can trigger asyncio fd cleanup errors in some runtimes.
os.environ.setdefault("GRADIO_HOT_RELOAD", "0")

warnings.filterwarnings(
    "ignore",
    message=r".*torch\.distributed\.reduce_op.*deprecated.*",
    category=FutureWarning,
)

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B"
DEFAULT_ADAPTER_PATH = "experiments/with_defs_qwen2_5_1_5B"
FALLBACK_ADAPTER_PATH = "experiments/qwen2_5_1_5B_masked_tuned"

DEFAULT_PREDICTION_EXPORT = "data/processed/active_learning/predictions_export.jsonl"
DEFAULT_CORRECTION_EXPORT = "data/processed/corrections/corrections.jsonl"
DEFAULT_AL_EXPORT = "data/processed/active_learning/cycle_records.jsonl"
DEFAULT_DASHBOARD_HTML = "plots/index.html"
DEFAULT_DASHBOARD_BUILDER = "scripts/build_plotly_dashboard.py"
LABEL_COLORS = {
    "PER": "#3b82f6",
    "ORG": "#22c55e",
    "LOC": "#f59e0b",
    "MISC": "#a855f7",
}

APP_CSS = """
.gradio-container {
  --bg: #070a14;
  --surface: #0d1326;
  --surface-soft: #111a33;
  --line: #25355f;
  --text: #e6f8ff;
  --muted: #93b4d5;
  --accent: #00eaff;
  --accent-2: #ff2bd6;
  --ok: #1aff9c;
  --warn: #ff9d00;
  background:
    radial-gradient(1100px 520px at -8% -18%, rgba(0, 234, 255, 0.14), transparent 64%),
    radial-gradient(1000px 460px at 110% -12%, rgba(255, 43, 214, 0.14), transparent 63%),
    linear-gradient(180deg, #050811 0%, #0a1020 100%);
  color: var(--text);
}
.gradio-container, .gradio-container * {
  font-family: "Rajdhani", "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif !important;
}
.app-wrap {max-width: 1260px; margin: 0 auto; padding: 6px 0 16px;}
.hero {
  margin-top: 8px;
  border: 1px solid var(--line);
  background: linear-gradient(125deg, #0b1122 0%, #101a35 58%, #121f3b 100%);
  color: var(--text);
  border-radius: 16px;
  padding: 18px 20px;
  box-shadow: 0 0 0 1px rgba(0,234,255,0.18), 0 16px 34px rgba(0, 0, 0, 0.45);
}
.hero h1 {
  margin: 0;
  font-size: 2.04rem;
  letter-spacing: 0.8px;
  font-weight: 840;
  color: #e8f7ff;
  text-shadow: 0 0 14px rgba(0, 234, 255, 0.35);
  line-height: 1.15;
}
.hero p {
  margin: 8px 0 0;
  color: var(--muted);
  font-size: 0.98rem;
  text-align: center;
}
.hero-top {
  display: grid;
  grid-template-columns: auto 1fr auto;
  align-items: center;
  column-gap: 10px;
}
.hero-title-wrap {
  text-align: center;
}
.hero-title-wrap h1 {
  text-align: center;
}
.hero-dash-btn {
  border: 1px solid #2a4e7f;
  background: linear-gradient(90deg, #0f1a34 0%, #13254a 100%);
  color: #bff5ff;
  border-radius: 10px;
  padding: 7px 10px;
  font-size: 0.82rem;
  font-weight: 700;
  white-space: nowrap;
  cursor: pointer;
  box-shadow: 0 0 0 1px rgba(0,234,255,0.24), 0 0 16px rgba(0,234,255,0.18);
}
.hero-dash-btn:hover {
  border-color: #33f0ff;
  background: linear-gradient(90deg, #112146 0%, #172d57 100%);
}
.hero-dash-spacer {
  width: 140px;
}
.hero-grid {
  margin-top: 13px;
  margin-left: auto;
  margin-right: auto;
  display: grid;
  grid-template-columns: repeat(2, minmax(130px, 1fr));
  gap: 9px;
}
.hero-chip {
  border: 1px solid #27416e;
  border-radius: 11px;
  padding: 8px 10px;
  background: #0d1730;
  color: #9fd0f8;
  font-weight: 700;
  font-size: 0.84rem;
}
.hero-chip strong {color: #19ecff;}
.panel-soft {
  border: 1px solid var(--line);
  border-radius: 16px;
  background: var(--surface);
  box-shadow: 0 10px 24px rgba(0, 0, 0, 0.38), 0 0 0 1px rgba(0,234,255,0.12);
  padding: 11px;
}
.dash-toolbar {
  border: 1px solid var(--line);
  border-radius: 12px;
  background: linear-gradient(90deg, #0f1931 0%, #101d39 100%);
  padding: 8px 11px;
}
.dash-note {
  margin-top: 8px;
  border: 1px dashed #2d4f7d;
  border-radius: 10px;
  background: #0d1630;
  color: #a4c7e6;
  padding: 10px 12px;
  font-size: 0.93rem;
  font-weight: 620;
  line-height: 1.45;
}
.dash-note code {
  color: #dffeff;
  background: #122447;
  border: 1px solid #2c5d93;
  border-radius: 6px;
  padding: 1px 6px;
}
.card {
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 13px;
  background: var(--surface);
  box-shadow: 0 10px 22px rgba(0,0,0,0.35), 0 0 0 1px rgba(0,234,255,0.08);
}
.card h3 {margin-bottom: 6px; color: var(--text);}
.meta {
  font-size: 0.95rem;
  color: #b4d6f2;
  line-height: 1.5;
  font-weight: 560;
}
.meta ul, .meta ol {margin: 0.2rem 0 0.25rem 0; padding-left: 1.1rem;}
.meta li {margin: 0.24rem 0; color: #b4d6f2;}
.meta li::marker {color: #00eaff;}
.meta strong {color: #58f4ff;}
.meta code {
  background: #12213f;
  color: #e2f8ff;
  border: 1px solid #2a4f7d;
  border-radius: 7px;
  padding: 1px 6px;
  font-weight: 700;
}
.legend {margin-top: 8px; display: flex; gap: 8px; flex-wrap: wrap;}
.pill {
  display: inline-flex;
  align-items: center;
  border: 1px solid #2d4f7a;
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 0.82rem;
  font-weight: 700;
  background: #0e1933;
  color: #b2d8f8;
}
.tips {
  border-radius: 10px;
  border: 1px dashed #295080;
  background: #0c1730;
  padding: 8px 11px;
  color: #9fc4e3;
  font-size: 0.88rem;
}
.hl-wrap {
  border: 1px solid #2a4e7a;
  border-radius: 12px;
  background: #0b1326;
  color: #def5ff;
  padding: 11px 12px;
  line-height: 1.6;
  font-size: 1.02rem;
  white-space: pre-wrap;
  word-break: break-word;
}
.ent {
  border-radius: 6px;
  padding: 2px 7px;
  margin: 0 1px;
  border: 1px solid transparent;
  font-weight: 700;
  color: #0f172a;
}
.ent small {margin-left: 5px; font-size: 0.68rem; opacity: .95; color: #334155;}
.ent small { color: #d7ecff; }
.card .prose, .card .prose * { color: #d9f6ff !important; }
.card .prose code {
  color: #d9f6ff !important;
  background: #122347 !important;
  border: 1px solid #2d5d93;
  border-radius: 6px;
  padding: 1px 6px;
}
.card .prose strong { color: #8af7ff !important; }
.gradio-button.primary {
  background: linear-gradient(90deg, var(--accent) 0%, var(--accent-2) 100%) !important;
  border: none !important;
  color: #031018 !important;
  font-weight: 750 !important;
  border-radius: 12px !important;
  box-shadow: 0 0 0 1px rgba(0,234,255,0.35), 0 0 20px rgba(0,234,255,0.4) !important;
}
.gradio-button.secondary {
  border: 1px solid #2a4f80 !important;
  background: #101a34 !important;
  color: #c9ecff !important;
  border-radius: 12px !important;
}
.tabs button {
  border-radius: 8px 0 0 0 !important;
  border: 1px solid #2a4f80 !important;
  background: #101a34 !important;
  color: #9dc8ee !important;
  font-weight: 650 !important;
  text-align: center !important;
  justify-content: center !important;
}
button:nth-child(2) {
    border-radius: 0 8px 0 0 !important;
}
.tabs button.selected {
  background: linear-gradient(90deg, #00eaff 0%, #1aff9c 100%) !important;
  color: #041018 !important;
  border-color: transparent !important;
  box-shadow: 0 0 0 1px rgba(0,234,255,0.3), 0 0 16px rgba(0,234,255,0.35) !important;
}
.gradio-container .tab-container {
  width: 100% !important;
  gap: 24px !important;
  text-align: center !important;
  justify-content: center !important;
  margin-left: auto !important;
  margin-right: auto !important;
}
.gradio-container .tab-container button {
  width: 48% !important;
  padding: 10px 20px !important;
  text-align: center !important;
  justify-content: center !important;
}
.status-box textarea {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  background: #0a1429 !important;
  color: #c9ecff !important;
  border: 1px solid #2a4f7e !important;
}
.card .cm-editor, .card pre, .card code {
  border-radius: 10px !important;
}
.card .cm-editor {
  border: 1px solid #2a4f7e !important;
  background: #0a1429 !important;
}
.card textarea, .card input, .card select {
  background: #0a1429 !important;
  color: #d5f3ff !important;
  border-color: #2a4f7e !important;
}
footer {
  margin-top: 10px;
  color: #84aed0;
  text-align: center;
  font-size: 0.82rem;
}
@media (max-width: 960px) {
  .hero-grid { grid-template-columns: 1fr; }
  .hero-top {
    grid-template-columns: 1fr;
    row-gap: 10px;
  }
  .hero-dash-spacer {
    display: none;
  }
  .hero-dash-btn {
    width: 100%;
  }
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Gradio correction interface.")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--adapter_path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--prediction_export", default=DEFAULT_PREDICTION_EXPORT)
    parser.add_argument("--correction_export", default=DEFAULT_CORRECTION_EXPORT)
    parser.add_argument("--active_learning_export", default=DEFAULT_AL_EXPORT)
    parser.add_argument("--dashboard_html_path", default=DEFAULT_DASHBOARD_HTML)
    parser.add_argument("--dashboard_builder", default=DEFAULT_DASHBOARD_BUILDER)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    return parser.parse_args()


def default_args() -> argparse.Namespace:
    """Default runtime args for environments that import `demo` directly (e.g., HF Spaces)."""
    port = int(os.getenv("PORT", "7860"))
    return SimpleNamespace(
        model_name=DEFAULT_MODEL_NAME,
        adapter_path=DEFAULT_ADAPTER_PATH,
        prediction_export=DEFAULT_PREDICTION_EXPORT,
        correction_export=DEFAULT_CORRECTION_EXPORT,
        active_learning_export=DEFAULT_AL_EXPORT,
        dashboard_html_path=DEFAULT_DASHBOARD_HTML,
        dashboard_builder=DEFAULT_DASHBOARD_BUILDER,
        host="0.0.0.0",
        port=port,
        share=False,
    )


def resolve_adapter(path: str) -> str | None:
    """
    Resolve adapter identifier.

    Supports:
    - Local filesystem paths
    - Hugging Face Hub repo IDs (e.g. `user/repo`)
    """
    if not path:
        return None

    if Path(path).exists():
        return path

    # Treat non-local repo-like identifiers as Hub adapter repos.
    if "/" in path and not path.startswith(".") and not path.startswith("/"):
        return path

    if Path(FALLBACK_ADAPTER_PATH).exists():
        return FALLBACK_ADAPTER_PATH

    print(
        f"Warning: adapter missing ({path}); fallback missing ({FALLBACK_ADAPTER_PATH}). "
        "Launching with base model only."
    )
    return None


def load_runtime(model_name: str, adapter_path: str | None):
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    applied_adapter: str | None = None
    if adapter_path:
        try:
            model = PeftModel.from_pretrained(model, adapter_path)
            applied_adapter = adapter_path
        except Exception as exc:
            print(f"Warning: failed to load adapter '{adapter_path}'. Reason: {exc}")
            print("Launching with base model only.")
    model.to(device)
    model.eval()
    return model, tokenizer, device, applied_adapter


def build_highlight_html(text: str, entities: list[dict]) -> str:
    if not text:
        return "<div class='hl-wrap'>Enter text and run extraction.</div>"

    spans: list[tuple[int, int, str]] = []
    for ent in entities:
        ent_text = (ent.get("text") or "").strip()
        label = (ent.get("label") or "").strip().upper()
        if not ent_text or label not in LABEL_COLORS:
            continue
        for m in re.finditer(re.escape(ent_text), text):
            spans.append((m.start(), m.end(), label))

    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    filtered: list[tuple[int, int, str]] = []
    last_end = -1
    for s, e, lbl in spans:
        if s >= last_end:
            filtered.append((s, e, lbl))
            last_end = e

    if not filtered:
        return f"<div class='hl-wrap'>{html_escape(text)}</div>"

    colors = {
        "PER": ("#0b2447", "#3b82f6", "#dbeafe"),
        "ORG": ("#102716", "#22c55e", "#dcfce7"),
        "LOC": ("#3b1b05", "#f59e0b", "#ffedd5"),
        "MISC": ("#2a1142", "#a855f7", "#f3e8ff"),
    }
    chunks = []
    cursor = 0
    for s, e, lbl in filtered:
        if cursor < s:
            chunks.append(html_escape(text[cursor:s]))
        bg, border, fg = colors.get(lbl, ("#0f172a", "#94a3b8", "#f8fafc"))
        ent_txt = html_escape(text[s:e])
        chunks.append(
            f"<span class='ent' style='background:{bg};border-color:{border};color:{fg};'>"
            f"{ent_txt}<small>{lbl}</small></span>"
        )
        cursor = e
    if cursor < len(text):
        chunks.append(html_escape(text[cursor:]))
    return "<div class='hl-wrap'>" + "".join(chunks) + "</div>"


def entity_stats_markdown(parsed: dict) -> str:
    entities = parsed.get("entities", [])
    counts = {"PER": 0, "ORG": 0, "LOC": 0, "MISC": 0}
    for ent in entities:
        lbl = (ent.get("label") or "").upper()
        if lbl in counts:
            counts[lbl] += 1
    total = len(entities)
    return (
        f"**Entity Count:** {total}  \n"
        f"- PER: `{counts['PER']}`  \n"
        f"- ORG: `{counts['ORG']}`  \n"
        f"- LOC: `{counts['LOC']}`  \n"
        f"- MISC: `{counts['MISC']}`"
    )


def make_predict_fn(model, tokenizer, device, state: CorrectionState):
    def predict(text: str):
        text = (text or "").strip()
        if not text:
            empty = json.dumps({"entities": []}, indent=2)
            return (
                empty,
                empty,
                "Enter text first.",
                "<div class='hl-wrap'>Enter text and run extraction.</div>",
                "**Entity Count:** 0",
                "",
            )

        prompt = build_prompt(text, prompt_style="with_defs")
        prompt += '\nReturn ONLY valid JSON with schema: {"entities":[{"text":"...","label":"PER|ORG|LOC|MISC"}]}\nOutput:\n'
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=192,
                do_sample=False,
                repetition_penalty=1.05,
                num_beams=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen = outputs[0][inputs["input_ids"].shape[1] :]
        decoded = tokenizer.decode(gen, skip_special_tokens=True)
        parsed = normalize_payload(extract_json(decoded) or {"entities": []})

        append_jsonl(
            state.prediction_export,
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "text": text,
                "raw_model_output": decoded,
                "prediction": parsed,
            },
        )

        state.processed_count += 1
        pretty = json.dumps(parsed, ensure_ascii=False, indent=2)
        state.last_message = f"Prediction saved: {state.prediction_export}"
        return (
            pretty,
            pretty,
            state.last_message,
            build_highlight_html(text, parsed.get("entities", [])),
            entity_stats_markdown(parsed),
            text,
        )

    return predict


def make_save_fn(state: CorrectionState):
    def save(text: str, predicted_json: str, corrected_json: str):
        text = (text or "").strip()
        if not text:
            return "Cannot save: text is empty."

        if not predicted_json:
            return "Cannot save: predicted JSON is empty. Run extraction first."
        if not corrected_json:
            return "Cannot save: corrected JSON is empty."

        try:
            predicted = normalize_payload(json.loads(predicted_json))
            corrected = normalize_payload(json.loads(corrected_json))
        except (json.JSONDecodeError, TypeError) as e:
            return f"Cannot save: invalid JSON ({e})"

        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "text": text,
            "predicted": predicted,
            "corrected": corrected,
            "accepted": predicted == corrected,
        }
        append_jsonl(state.correction_export, record)
        append_cycle_record(state.active_learning_export, build_cycle_record(text, predicted, corrected))

        state.saved_count += 1
        state.last_message = (
            f"Saved correction #{state.saved_count} -> {state.correction_export}; "
            f"active learning -> {state.active_learning_export}"
        )
        return state.last_message

    return save


def render_dashboard_iframe(dashboard_html_path: str) -> str:
    path = Path(dashboard_html_path)
    if not path.exists():
        return (
            "<div class='dash-note'>Dashboard file not found: "
            f"<code>{dashboard_html_path}</code>. Run "
            "<code>python scripts/build_plotly_dashboard.py</code> first.</div>"
        )
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return (
        "<iframe "
        "title='Experiment Dashboard' "
        "src='data:text/html;base64,"
        + encoded
        + "' "
        "style='width:100%;height:100%;border:1px solid #275181;border-radius:12px;background:#0a1328;' "
        "loading='lazy'></iframe>"
    )


def render_dashboard_overlay(dashboard_html_path: str, visible: bool) -> str:
    display = "flex" if visible else "none"
    iframe = render_dashboard_iframe(dashboard_html_path)
    return f"""
    <div id="analytics-overlay" style="
      position:fixed; inset:0; z-index:9999; display:{display};
      align-items:center; justify-content:center; background:rgba(2,6,23,0.68);
      backdrop-filter:blur(2px);">
      <div style="
        width:min(1420px, 96vw); height:min(940px, 95vh);
        background:#0a1328; border-radius:14px; box-shadow:0 24px 60px rgba(2,6,23,0.55);
        border:1px solid #275181; display:flex; flex-direction:column; overflow:hidden;">
        <div style="
          padding:8px 12px; border-bottom:1px solid #275181; background:#0f1d3b;
          display:flex; align-items:center; justify-content:space-between;">
          <div style="font-weight:700; color:#dff8ff; letter-spacing:.4px;">Analytics Dashboard</div>
          <button onclick="document.getElementById('analytics-overlay').style.display='none'"
            style="
              border:1px solid #2c5f94; background:#102247; color:#dff8ff;
              border-radius:8px; padding:6px 10px; font-weight:600; cursor:pointer;">
            Close
          </button>
        </div>
        <div style="padding:8px; height:calc(100% - 46px); background:#0a1328;">
          {iframe}
        </div>
      </div>
    </div>
    """


def build_demo(args: argparse.Namespace):
    """Build and return Gradio demo + runtime metadata."""
    adapter_path = resolve_adapter(args.adapter_path)
    model, tokenizer, device, applied_adapter = load_runtime(args.model_name, adapter_path)
    adapter_display = applied_adapter or "base-model-only (adapter not loaded)"

    state = CorrectionState(
        model_name=args.model_name,
        adapter_path=adapter_display,
        prediction_export=args.prediction_export,
        correction_export=args.correction_export,
        active_learning_export=args.active_learning_export,
    )

    predict_fn = make_predict_fn(model, tokenizer, device, state)
    save_fn = make_save_fn(state)
    with gr.Blocks(title="NER JSON Studio") as demo:
        with gr.Column(elem_classes=["app-wrap"]):
            gr.HTML(
                """
                <div class='hero'>
                  <div class='hero-top'>
                    <button class='hero-dash-btn' onclick="document.getElementById('analytics-overlay').style.display='flex'">
                      Analytics Dashboard
                    </button>
                    <div class='hero-title-wrap'>
                      <h1>NER to JSON Studio</h1>
                      <p>NER JSON Studio | Fine-tuned Qwen2.5-1.5B (LoRA) | F1: 90.48% | JSON Validity: 100%</p>
                    </div>
                    <div class='hero-dash-spacer'></div>
                  </div>
                </div>
                """
            )
            with gr.Tabs():
                with gr.Tab("Extract"):
                    with gr.Row():
                        with gr.Column(scale=2, elem_classes=["card"]):
                            gr.Markdown("### Input Text")
                            input_text = gr.Textbox(
                                label="Raw Text",
                                lines=8,
                                placeholder="Paste sentence or paragraph here...",
                            )
                            gr.HTML("<div class='tips'>Tip: Use clear sentence boundaries for better entity extraction.</div>")
                            gr.Examples(
                                examples=[
                                    ["Barack Obama visited Paris during his Europe tour."],
                                    ["Apple announced new products in California with Tim Cook on stage."],
                                ],
                                inputs=input_text,
                            )
                            with gr.Row():
                                predict_btn = gr.Button("Extract Entities", variant="primary")
                                clear_btn = gr.Button("Clear", variant="secondary")

                        with gr.Column(scale=1, elem_classes=["card"]):
                            gr.Markdown("### Model Snapshot")
                            gr.Markdown(
                                "\n".join(
                                    [
                                        f"- **Device:** `{device}`",
                                        f"- **Model:** `{args.model_name}`",
                                        f"- **Adapter:** `{adapter_display}`",
                                        "- **Best Config:** `json_validate=yes`, `temperature=0.1`, `constrained`",
                                        "- **Reference:** `F1 ~ 0.90+`, `JSON validity 100%`",
                                    ]
                                ),
                                elem_classes=["meta"],
                            )
                            gr.HTML(
                                """
                                <div class='legend'>
                                  <span class='pill'>PER: Blue</span>
                                  <span class='pill'>ORG: Green</span>
                                  <span class='pill'>LOC: Amber</span>
                                  <span class='pill'>MISC: Violet</span>
                                </div>
                                """
                            )

                    with gr.Row():
                        with gr.Column(elem_classes=["card"]):
                            gr.Markdown("### Extracted JSON")
                            predicted_json = gr.Code(label="Predicted JSON", language="json", interactive=False)
                        with gr.Column(elem_classes=["card"]):
                            gr.Markdown("### Entity Stats")
                            entity_stats = gr.Markdown("**Entity Count:** 0")

                    with gr.Row():
                        with gr.Column(elem_classes=["card"]):
                            gr.Markdown("### Highlighted Entities")
                            highlighted_text = gr.HTML("<div class='hl-wrap'>Enter text and run extraction.</div>")

                    status = gr.Textbox(label="Extract Status", interactive=False, elem_classes=["status-box"])

                with gr.Tab("Correct & Save"):
                    with gr.Row():
                        with gr.Column(scale=2, elem_classes=["card"]):
                            gr.Markdown("### Review Text")
                            text_for_correction = gr.Textbox(
                                label="Source Text (copied from Extract tab)",
                                lines=6,
                            )
                        with gr.Column(elem_classes=["card"]):
                            gr.Markdown("### Correction JSON")
                            corrected_json = gr.Code(label="Corrected JSON (Editable)", language="json")

                    with gr.Row():
                        save_btn = gr.Button("Approve / Save Correction", variant="primary")

                    correction_status = gr.Textbox(label="Correction Status", interactive=False, elem_classes=["status-box"])

                    with gr.Column(elem_classes=["card"]):
                        gr.Markdown("### Export Targets")
                        gr.Markdown(
                            "\n".join(
                                [
                                    f"- **Predictions:** `{state.prediction_export}`",
                                    f"- **Corrections:** `{state.correction_export}`",
                                    f"- **Active Learning:** `{state.active_learning_export}`",
                                ]
                            ),
                            elem_classes=["meta"],
                        )
            dashboard_overlay = gr.HTML(value=render_dashboard_overlay(args.dashboard_html_path, visible=False))

            predict_btn.click(
                predict_fn,
                inputs=[input_text],
                outputs=[predicted_json, corrected_json, status, highlighted_text, entity_stats, text_for_correction],
            )
            save_btn.click(
                save_fn,
                inputs=[text_for_correction, predicted_json, corrected_json],
                outputs=[correction_status],
            )
            clear_btn.click(
                lambda: (
                    "",
                    "",
                    "",
                    "Cleared.",
                    "<div class='hl-wrap'>Enter text and run extraction.</div>",
                    "**Entity Count:** 0",
                    "",
                    "Cleared.",
                ),
                outputs=[
                    input_text,
                    predicted_json,
                    corrected_json,
                    status,
                    highlighted_text,
                    entity_stats,
                    text_for_correction,
                    correction_status,
                ],
            )
            gr.HTML("<footer>Built with Gradio | LoRA Fine-tuned | CoNLL2003 Dataset</footer>")

    return demo, device, adapter_display


def main() -> None:
    args = parse_args()
    demo, device, adapter_display = build_demo(args)
    print(f"Device: {device}")
    print(f"Adapter: {adapter_display}")
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        css=APP_CSS,
        theme=gr.themes.Soft(),
        ssr_mode=False,
        quiet=False,
    )


if __name__ == "__main__":
    main()
