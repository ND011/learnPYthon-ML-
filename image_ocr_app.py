import streamlit as st
from transformers import pipeline
import torch
import easyocr
from PIL import Image
import tempfile
import os
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Configure Streamlit page
st.set_page_config(page_title="Image OCR App", layout="wide")

# App Layout
st.title("🖼️ Image OCR & Analysis Application")
st.write("Extract and analyze text from images")

# Load models
@st.cache_resource(ttl=3600)
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_path = r"C:/0ND/Project/models/modelchash"
    
    def load_model_with_fallback(model_type, local_dir, repo_id, **kwargs):
        try:
            path = os.path.join(base_path, local_dir)
            if os.path.exists(path):
                return pipeline(model_type, model=path, **kwargs)
            else:
                return pipeline(model_type, model=repo_id, **kwargs)
        except Exception as e:
            st.warning(f"❌ Failed to load {model_type}: {str(e)}")
            return None

    # Load summarizer
    summarizer = load_model_with_fallback(
        "summarization",
        "pegasus-xsum",
        "google/pegasus-xsum",
        device=device
    )
    
    # Load OCR reader
    reader = easyocr.Reader(['en'])
    
    return summarizer, reader

# Load models
with st.spinner("🔄 Loading AI models..."):
    summarizer, reader = load_models()
st.success("✅ Models loaded successfully")

# Image upload
image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

if image_file:
    try:
        # Debug information
        st.write(f"Debug - File name: {image_file.name}")
        st.write(f"Debug - File type: {image_file.type}")
        
        # Get file extension and validate
        file_extension = image_file.name.split('.')[-1].lower() if '.' in image_file.name else ''
        if not file_extension or file_extension not in ['jpg', 'jpeg', 'png']:
            st.error("Please upload a valid image file (JPG, JPEG, or PNG)")
            st.stop()
        
        image = Image.open(image_file)
        # Resize image to fixed dimensions while maintaining aspect ratio
        max_width = 800  # You can adjust this value
        max_height = 600  # You can adjust this value
        
        # Calculate new dimensions maintaining aspect ratio
        img_width, img_height = image.size
        aspect_ratio = img_width / img_height
        
        if img_width > max_width:
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        elif img_height > max_height:
            new_height = max_height
            new_width = int(max_height * aspect_ratio)
        else:
            new_width = img_width
            new_height = img_height
            
        # Resize image
        image_resized = image.resize((new_width, new_height))
        # Display image with fixed size
        st.image(image_resized, caption='Uploaded Image', use_column_width=False)
        
        if st.button("Extract Text & Generate Summary"):
            with st.spinner("Processing..."):
                # Save image temporarily with explicit extension
                with tempfile.NamedTemporaryFile(suffix=f'.{file_extension}', delete=False) as tmp_file:
                    img_path = tmp_file.name
                    # Convert image to RGB if needed
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    # Save with explicit format
                    save_format = 'PNG' if file_extension == 'png' else 'JPEG'
                    image.save(img_path, format=save_format)
                
                # Extract text using OCR
                result = reader.readtext(img_path, detail=0)
                extracted_text = ' '.join(result)
                
                if not extracted_text.strip():
                    st.warning("No text was detected in the image.")
                    os.unlink(img_path)
                    st.stop()
                
                # Generate summary if text is long enough
                if len(extracted_text.split()) > 30:  # Only summarize if there's enough text
                    summary = summarizer(extracted_text, max_length=300, min_length=30, do_sample=False)[0]['summary_text']
                else:
                    summary = "Text is too short to generate a meaningful summary."
                
                # Extract keywords
                if len(extracted_text.split()) > 5:  # Only extract keywords if there's enough text
                    vec = TfidfVectorizer(stop_words='english', max_features=1000)
                    tfidf_matrix = vec.fit_transform([extracted_text])
                    scores = zip(vec.get_feature_names_out(), tfidf_matrix.toarray()[0])
                    sorted_words = sorted(scores, key=lambda x: x[1], reverse=True)
                    keywords = [word for word, score in sorted_words[:10]]
                else:
                    keywords = []
                
                # Display results
                st.subheader("📝 Extracted Text")
                st.text_area("Text", extracted_text, height=150)
                
                st.subheader("📄 Summary")
                st.info(summary)
                
                if keywords:
                    st.subheader("🔑 Extracted Keywords")
                    st.markdown(f"**Top Keywords:** {', '.join(keywords)}")
                    
                    # Generate word cloud with fixed size
                    st.subheader("☁️ Word Cloud")
                    wordcloud = WordCloud(width=800, height=300, background_color='white').generate(' '.join(keywords))
                    # Convert wordcloud to image and resize
                    wordcloud_image = Image.fromarray(wordcloud.to_array())
                    wordcloud_resized = wordcloud_image.resize((800, 300))
                    st.image(wordcloud_resized, use_column_width=False)
                
                # Download functionality
                def generate_download_link(text, filename="extracted_text.txt"):
                    b64 = base64.b64encode(text.encode()).decode()
                    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">📥 Download Extracted Text</a>'
                    return href
                
                st.markdown(generate_download_link(extracted_text), unsafe_allow_html=True)
                
                # Clean up temporary file
                os.unlink(img_path)
                
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.info("Please try uploading a different image or ensure the image format is supported.")

# Footer
st.markdown("---")
st.markdown("Created by Dhaval Thaker | Image OCR & Analysis Application") 