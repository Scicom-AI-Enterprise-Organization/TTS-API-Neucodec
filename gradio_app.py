"""
Simple Gradio frontend to smoke-test a running TTS-API instance (app/main.py).

Run with:
    python gradio_app.py
Then open the printed local URL and set "API base URL" to wherever the API is
reachable, e.g. http://localhost:9091 or a remote/tunnelled address.
"""

import os
import tempfile

import gradio as gr
import requests

DEFAULT_BASE_URL = os.environ.get("TTS_API_BASE_URL", "http://localhost:9091")


def _save_wav(content, name):
    # Use tempfile.gettempdir() rather than a hardcoded "/tmp": on macOS "/tmp" is a symlink to
    # "/var/folders/.../T", and Gradio's file-serving safety check rejects paths outside the
    # resolved temp dir it actually expects.
    path = os.path.join(tempfile.gettempdir(), name)
    with open(path, "wb") as f:
        f.write(content)
    return path


def check_connection(base_url):
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/v1/audio/speaker", timeout=10)
        resp.raise_for_status()
        speakers = resp.json()
        return f"✅ Connected. Speakers: {speakers}", gr.update(choices=speakers, value=speakers[0] if speakers else None)
    except Exception as e:
        return f"❌ Could not reach {base_url}: {e}", gr.update()


def text_to_speech(base_url, text, voice, normalize_malaysian, temperature, playback_speed):
    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/v1/audio/speech",
            json={
                "input": text,
                "voice": voice,
                "normalize_malaysian": normalize_malaysian,
                "temperature": temperature,
                "playback_speed": playback_speed,
                "response_format": "wav",
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return _save_wav(resp.content, "tts_api_test_speech.wav"), "✅ Generated."
    except Exception as e:
        return None, f"❌ Request failed: {e}"


def normalize_text(base_url, text, normalize_malaysian):
    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/v1/audio/normalize",
            json={"input": text, "normalize_malaysian": normalize_malaysian},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("output", ""), "✅ Normalized."
    except Exception as e:
        return "", f"❌ Request failed: {e}"


def voice_conversion(base_url, reference_audio_path, reference_text, generate_text, temperature, playback_speed):
    try:
        with open(reference_audio_path, "rb") as f:
            resp = requests.post(
                f"{base_url.rstrip('/')}/v1/audio/vc",
                files={"reference_audio": f},
                data={
                    "reference_text": reference_text,
                    "generate_text": generate_text,
                    "temperature": temperature,
                    "playback_speed": playback_speed,
                    "response_format": "wav",
                    "stream": False,
                },
                timeout=120,
            )
        resp.raise_for_status()
        return _save_wav(resp.content, "tts_api_test_vc.wav"), "✅ Generated."
    except Exception as e:
        return None, f"❌ Request failed: {e}"


with gr.Blocks(title="TTS-API tester") as demo:
    gr.Markdown("# TTS-API tester\nPoint this at any running instance of the API to check it's reachable and working.")

    base_url = gr.Textbox(label="API base URL", value=DEFAULT_BASE_URL)
    check_btn = gr.Button("Check connection")
    status = gr.Textbox(label="Status", interactive=False)

    with gr.Tab("Text to speech"):
        tts_text = gr.Textbox(label="Text", value="Hello there, how can I help you today?", lines=3)
        with gr.Row():
            tts_voice = gr.Dropdown(label="Voice", choices=[], allow_custom_value=True)
            tts_normalize = gr.Checkbox(label="normalize_malaysian", value=False)
        with gr.Row():
            tts_temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=1.5, value=0.7, step=0.05)
            tts_playback_speed = gr.Slider(label="Playback speed", minimum=0.5, maximum=4.0, value=1.5, step=0.1)
        tts_btn = gr.Button("Generate speech", variant="primary")
        tts_status = gr.Textbox(label="Status", interactive=False)
        tts_audio = gr.Audio(label="Output", type="filepath", show_download_button=True)

        tts_btn.click(
            text_to_speech,
            inputs=[base_url, tts_text, tts_voice, tts_normalize, tts_temperature, tts_playback_speed],
            outputs=[tts_audio, tts_status],
        )

    with gr.Tab("Normalize text"):
        norm_text = gr.Textbox(label="Text", value="价格是RM500 电话012-1234567", lines=3)
        norm_malaysian = gr.Checkbox(label="normalize_malaysian", value=True)
        norm_btn = gr.Button("Normalize", variant="primary")
        norm_status = gr.Textbox(label="Status", interactive=False)
        norm_output = gr.Textbox(label="Normalized output", lines=3)

        norm_btn.click(
            normalize_text,
            inputs=[base_url, norm_text, norm_malaysian],
            outputs=[norm_output, norm_status],
        )

    with gr.Tab("Voice conversion"):
        vc_reference_audio = gr.Audio(label="Reference audio", type="filepath")
        vc_reference_text = gr.Textbox(label="Reference text (transcript of the reference audio)")
        vc_generate_text = gr.Textbox(label="Text to generate", lines=3)
        with gr.Row():
            vc_temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=1.5, value=0.7, step=0.05)
            vc_playback_speed = gr.Slider(label="Playback speed", minimum=0.5, maximum=4.0, value=1.5, step=0.1)
        vc_btn = gr.Button("Generate", variant="primary")
        vc_status = gr.Textbox(label="Status", interactive=False)
        vc_audio = gr.Audio(label="Output", type="filepath", show_download_button=True)

        vc_btn.click(
            voice_conversion,
            inputs=[base_url, vc_reference_audio, vc_reference_text, vc_generate_text, vc_temperature, vc_playback_speed],
            outputs=[vc_audio, vc_status],
        )

    check_btn.click(check_connection, inputs=[base_url], outputs=[status, tts_voice])

if __name__ == "__main__":
    demo.launch()
