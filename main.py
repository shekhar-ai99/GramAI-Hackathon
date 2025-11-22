# main.py
import gradio as gr
from models.paddy_model import predict_paddy
from models.skin_model import predict_skin
from utils.tts import speak_odia

# Beautiful Odisha background + styling
css = """
body { 
    background: url('https://i.imgur.com/8Y6z1kT.jpg') no-repeat center center fixed; 
    background-size: cover; 
    font-family: 'Segoe UI', sans-serif;
}
.container { 
    background: rgba(255,255,255,0.96); 
    border-radius: 25px; 
    padding: 30px; 
    box-shadow: 0 10px 40px rgba(0,0,0,0.3); 
    margin: 20px;
}
h1 { 
    font-family: 'Noto Sans Oriya', sans-serif; 
    color: #006400; 
    text-align: center; 
    font-size: 52px; 
    text-shadow: 2px 2px 8px rgba(0,0,0,0.2);
}
"""

with gr.Blocks(css=css, title="GramAI – ଗ୍ରାମଏଆଇ") as demo:
    gr.HTML("""
    <div class="container">
        <h1>ଗ୍ରାମଏଆଇ – GramAI</h1>
        <p style="text-align:center; font-size:24px; color:#006400;">
            <b>ଓଡ଼ିଶାର କୃଷକଙ୍କ ପାଇଁ ପ୍ରଥମ AI ସାଥୀ</b><br>
            ଧାନ ରୋଗ + ଚର୍ମ ରୋଗ → ଗୋଟିଏ ଫଟୋରେ ଚିହ୍ନଟ + ଓଡ଼ିଆ ଧ୍ୱନି
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            mode = gr.Radio(
                ["Paddy / ଧାନ", "Skin / ଚର୍ମ"],
                label="କେଉଁଟି ଦେଖିବେ?",
                value="Paddy / ଧାନ",
                elem_classes="container"
            )
            img_input = gr.Image(type="pil", label="ଫଟୋ ଅପଲୋଡ୍ କରନ୍ତୁ", height=520)

        with gr.Column(scale=1):
            gr.HTML('<div class="container">')
            btn = gr.Button("ଚିହ୍ନଟ କରନ୍ତୁ", variant="primary", size="lg")
            result = gr.Textbox(label="ଫଳାଫଳ", lines=8, elem_classes="container")
            audio = gr.Audio(label="ଓଡ଼ିଆରେ ଶୁଣନ୍ତୁ")
            gr.HTML('</div>')

    def predict(img, mode):
        if "ଧାନ" in mode or "Paddy" in mode:
            text = predict_paddy(img)
        else:
            text = predict_skin(img)
        audio_path = speak_odia(text)
        return text, audio_path

    btn.click(predict, [img_input, mode], [result, audio])

    gr.HTML("""
    <center>
        <p style="margin-top:40px; font-size:18px; color:#006400;">
            Made with ❤️ for Northern Odisha Farmers<br>
            Balasore • Mayurbhanj • Keonjhar • Bhadrak • Angul
        </p>
    </center>
    """)

if __name__ == "__main__":
    demo.launch(share=True)