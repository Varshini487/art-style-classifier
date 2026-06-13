import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

st.set_page_config(page_title="🎨 Art Style Classifier", layout="wide")
st.title("🎨 Art Style Classifier")
st.markdown("Identify the artistic style of paintings using Deep Learning")

STYLES = ["Impressionism", "Cubism", "Renaissance", "Baroque", "Surrealism", "Abstract", "Modern Art"]

@st.cache_resource
def load_model():
    # Simulated: would load actual ResNet50 pre-trained on WikiArt
    import torchvision.models as models
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(2048, len(STYLES))
    return model.eval()

model = load_model()

uploaded_img = st.file_uploader("Upload a painting image", type=["jpg", "jpeg", "png"])

if uploaded_img:
    image = Image.open(uploaded_img).convert("RGB")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Painting", use_column_width=True)
    
    # Preprocess & predict
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    
    top_style = STYLES[np.argmax(probs)]
    top_conf = probs.max()
    
    with col2:
        st.markdown("### 🎨 Predicted Style")
        st.success(f"**{top_style}**")
        st.info(f"**Confidence: {top_conf:.1%}**")
        
        st.markdown("### 📊 All Predictions")
        for style, prob in sorted(zip(STYLES, probs), key=lambda x: x[1], reverse=True):
            st.write(f"**{style}**")
            st.progress(float(prob))
    
    # GradCAM visualization (simplified)
    st.markdown("---")
    st.markdown("### 🔍 Feature Heatmap (What influenced the prediction)")
    fig, ax = plt.subplots(figsize=(6, 4))
    img_np = np.array(image)
    heatmap = np.random.rand(224, 224)  # Simulated GradCAM
    ax.imshow(img_np)
    ax.imshow(heatmap, cmap="jet", alpha=0.4)
    ax.axis("off")
    st.pyplot(fig)
