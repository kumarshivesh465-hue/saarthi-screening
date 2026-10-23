import os
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model('saarthi_prototype.keras')

CLASSES = ['Normal', 'Refer', 'Urgent']
COLORS = {'Normal': '#10B981', 'Refer': '#F59E0B', 'Urgent': '#EF4444'}
REFERRALS = {
    'Normal': 'Routine follow-up in 12 months.',
    'Refer': 'Refer to ophthalmologist within 14 days.',
    'Urgent': 'Immediate referral — same-week appointment.'
}

def predict(img):
    img = img.resize((224, 224))
    arr = np.array(img)
    arr = tf.keras.applications.mobilenet_v3.preprocess_input(arr)
    pred = model.predict(np.expand_dims(arr, axis=0))[0]
    
    if pred[2] > 0.30:
        top_idx = 2  # Urgent
    else:
        top_idx = np.argmax(pred)
        
    top_class = CLASSES[top_idx]
    confidence = float(pred[top_idx]) * 100
    
    banner = f"""
    <div style='background-color: {COLORS[top_class]}; padding: 24px; border-radius: 12px; text-align: center; color: white;'>
        <h1 style='margin: 0; font-size: 32px;'>{top_class}</h1>
        <p style='margin: 5px 0 0 0; font-size: 18px;'>Confidence: {confidence:.1f}%</p>
    </div>
    """
    
    conf_dict = {CLASSES[i]: float(pred[i]) for i in range(3)}
    referral_text = REFERRALS[top_class]
    
    return banner, conf_dict, referral_text

custom_css = """
body { background-color: #f8fafc; }
.gradio-container { max-width: 1100px !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), css=custom_css) as demo:
    gr.Markdown("# 🏥 Saarthi: AI Retinal Screening")
    gr.Markdown("Upload a retinal fundus image. The AI will triage it as **Normal**, **Refer**, or **Urgent**.")
    
    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Retinal Image")
            submit_btn = gr.Button("🔬 Run Screening", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            result_banner = gr.HTML(label="Result")
            conf_bar = gr.Label(label="Confidence Scores", num_top_classes=3)
            referral_box = gr.Textbox(label="Referral Recommendation", interactive=False)
            
    submit_btn.click(fn=predict, inputs=img_input, outputs=[result_banner, conf_bar, referral_box])
    
    gr.Markdown("---\n*Research prototype. Not a medical device. Not for clinical use.*")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
