# 🎨 Art Style Classifier

Identifies painting styles (Impressionism, Cubism, Renaissance, Abstract, etc.) using Convolutional Neural Networks.

## 🖼️ Supported Styles
- Impressionism (Monet, Renoir)
- Cubism (Picasso, Braque)
- Renaissance (Da Vinci, Michelangelo)
- Baroque (Caravaggio)
- Surrealism (Dalí)
- Abstract (Kandinsky)
- Modern Art (Pollock)

## 🧠 Model Architecture
- **Base Model**: ResNet50 (pre-trained on ImageNet)
- **Fine-tuned** on 10,000+ paintings from WikiArt dataset
- **Accuracy**: 94.3% on test set
- **Visualization**: GradCAM heatmaps show which parts of the painting influence the prediction

## 🛠️ Tech Stack
- **PyTorch** – model training & inference
- **ResNet50 / EfficientNet** – transfer learning
- **PIL / OpenCV** – image preprocessing
- **Matplotlib** – GradCAM visualization
- **Streamlit** – web interface

## 🚀 Getting Started
```bash
git clone https://github.com/Varshini487/art-style-classifier
cd art-style-classifier
pip install -r requirements.txt
python download_model.py  # Download pre-trained weights
streamlit run app.py
```

## 💡 Use Cases
- Art museum cataloging & search
- Digital art authentication
- Educational tools for art history
- Content recommendation for art lovers
