# main.py — FINAL WINNING VERSION (Mobile + Tablet Responsive + Local BG)
import gradio as gr
from models.paddy_model import predict_paddy
from models.skin_model import predict_skin
from utils.tts import speak_odia

# FULLY RESPONSIVE + LOCAL BACKGROUND + CLEAN MOBILE VIEW
css = """
<style>
    body {
        background: url('./assets/odisha_bg.jpg') no-repeat center center fixed;
        background-size: cover;
        font-family: 'Segoe UI', 'Noto Sans Oriya', sans-serif;
        margin: 0;
        padding: 10px;
    }
    .container {
        background: rgba(255, 255, 255, 0.96);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
        margin: 15px auto;
        max-width: 900px;
    }
    h1 {
        font-family: 'Noto Sans Oriya', sans-serif;
        color: #006400;
        text-align: center;
        font-size: clamp(32px, 8vw, 56px);
        margin: 10px 0;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.2);
    }
    .gradio-container {
        max-width: 1000px !important;
        margin: auto;
    }
    @media (max-width: 768px) {
        .gr-form { flex-direction: column !important; }
        .gr-button { width: 100%; margin-top: 15px; }
        .gr-image, .gr-audio { width: 100% !important; }
    }
</style>
"""

with gr.Blocks(css=css, title="GramAI – ଗ୍ରାମଏଆଇ") as demo:
    gr.HTML("""
    <div class="container">
        <h1>ଗ୍ରାମଏଆଇ – GramAI</h1>
        <p style="text-align:center; font-size: clamp(18px, 5vw, 24px); color:#006400; margin:10px 0;">
            <b>ଓଡ଼ିଶାର କୃଷକଙ୍କ ପାଇଁ ପ୍ରଥମ AI ସାଥୀ</b><br>
            ଧାନ ରୋଗ + ଚର୍ମ ରୋଗ → ଗୋଟିଏ ଫଟୋରେ ଚିହ୍ନଟ + ଓଡ଼ିଆ ଧ୍ୱନି
        </p>
    </div>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=300):
            mode = gr.Radio(
                ["Paddy / ଧାନ", "Skin / ଚର୍ମ"],
                label="କେଉଁଟି ଦେଖିବେ?",
                value="Paddy / ଧାନ",
                elem_classes="container"
            )
            img_input = gr.Image(
                type="pil",
                label="ଫଟୋ ଅପଲୋଡ୍ କରନ୍ତୁ",
                height=450,
                elem_classes="container"
            )

        with gr.Column(scale=1, min_width=300):
            gr.HTML('<div class="container">')
            btn = gr.Button(
                "ଚିହ୍ନଟ କରନ୍ତୁ",
                variant="primary",
                size="lg"
            )
            result = gr.Textbox(
                label="ଫଳାଫଳ",
                lines=8,
                elem_classes="container"
            )
            audio = gr.Audio(
                label="ଓଡ଼ିଆରେ ଶୁଣନ୍ତୁ",
                elem_classes="container"
            )
            gr.HTML('</div>')

    def predict(img, mode):
        if "ଧାନ" in mode or "Paddy" in mode:
            text = predict_paddy(img)
        else:
            text = predict_skin(img)
        audio_path = speak_odia(text)
        return text, audio_path

    btn.click(predict, inputs=[img_input, mode], outputs=[result, audio])

    gr.HTML("""
    <div class="container" style="text-align:center; margin-top:30px;">
        <p style="font-size:18px; color:#006400;">
            Made with ❤️ for Northern Odisha Farmers<br>
            Balasore • Mayurbhanj • Keonjhar • Bhadrak • Angul
        </p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch(share=True)