#!/usr/bin/env python3
"""
Test Applications - Simple working versions
"""

import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import base64

# Configure Streamlit
st.set_page_config(page_title="Test AI Apps", layout="wide")

# App Layout
st.title("🧪 Test AI Applications")
st.write("Simple working versions of AI features")

# Sidebar
st.sidebar.title("🔧 Select Feature")
option = st.sidebar.radio("Choose a feature:", [
    "Text Statistics",
    "Data Analysis", 
    "Simple Visualization",
    "File Processing"
])

if option == "Text Statistics":
    st.subheader("📝 Text Statistics")
    
    text_input = st.text_area("Enter text to analyze:", height=200)
    
    if st.button("Analyze Text"):
        if text_input.strip():
            # Basic statistics
            words = text_input.split()
            sentences = text_input.split('.')
            paragraphs = text_input.split('\n\n')
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Words", len(words))
            with col2:
                st.metric("Characters", len(text_input))
            with col3:
                st.metric("Sentences", len([s for s in sentences if s.strip()]))
            with col4:
                st.metric("Paragraphs", len([p for p in paragraphs if p.strip()]))
            
            # Word frequency
            word_freq = Counter(word.lower().strip('.,!?";:()[]') for word in words if len(word) > 2)
            
            st.subheader("🔑 Most Common Words")
            for word, count in word_freq.most_common(10):
                st.write(f"**{word}**: {count} times")
            
            # Character analysis
            st.subheader("📊 Character Analysis")
            char_counts = {
                'Letters': sum(c.isalpha() for c in text_input),
                'Numbers': sum(c.isdigit() for c in text_input),
                'Spaces': sum(c.isspace() for c in text_input),
                'Punctuation': sum(not c.isalnum() and not c.isspace() for c in text_input)
            }
            
            for char_type, count in char_counts.items():
                st.write(f"**{char_type}**: {count}")
        else:
            st.warning("Please enter some text first.")

elif option == "Data Analysis":
    st.subheader("📊 Data Analysis")
    
    # Sample data generation
    if st.button("Generate Sample Data"):
        # Create sample dataset
        np.random.seed(42)
        data = {
            'Name': [f'Item_{i}' for i in range(1, 101)],
            'Value': np.random.normal(100, 20, 100),
            'Category': np.random.choice(['A', 'B', 'C', 'D'], 100),
            'Score': np.random.uniform(0, 100, 100)
        }
        df = pd.DataFrame(data)
        
        # Display data
        st.subheader("📋 Sample Dataset")
        st.dataframe(df.head(10))
        
        # Basic statistics
        st.subheader("📈 Statistics")
        st.write(df.describe())
        
        # Category counts
        st.subheader("📊 Category Distribution")
        category_counts = df['Category'].value_counts()
        st.bar_chart(category_counts)
        
        # Download link
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="sample_data.csv">📥 Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

elif option == "Simple Visualization":
    st.subheader("📈 Simple Visualization")
    
    # Data input
    col1, col2 = st.columns(2)
    with col1:
        x_values = st.text_input("X values (comma-separated):", "1,2,3,4,5")
    with col2:
        y_values = st.text_input("Y values (comma-separated):", "2,4,6,8,10")
    
    if st.button("Create Plot"):
        try:
            x = [float(val.strip()) for val in x_values.split(',')]
            y = [float(val.strip()) for val in y_values.split(',')]
            
            if len(x) == len(y):
                # Create plot
                fig, ax = plt.subplots()
                ax.plot(x, y, marker='o')
                ax.set_xlabel('X Values')
                ax.set_ylabel('Y Values')
                ax.set_title('Simple Line Plot')
                ax.grid(True)
                
                st.pyplot(fig)
                
                # Statistics
                st.subheader("📊 Plot Statistics")
                st.write(f"**Points**: {len(x)}")
                st.write(f"**X Range**: {min(x)} to {max(x)}")
                st.write(f"**Y Range**: {min(y)} to {max(y)}")
            else:
                st.error("X and Y must have the same number of values")
        except ValueError:
            st.error("Please enter valid numbers separated by commas")

elif option == "File Processing":
    st.subheader("📁 File Processing")
    
    uploaded_file = st.file_uploader("Choose a file", type=['txt', 'csv'])
    
    if uploaded_file is not None:
        # File info
        st.write(f"**Filename**: {uploaded_file.name}")
        st.write(f"**File size**: {uploaded_file.size} bytes")
        
        if uploaded_file.name.endswith('.txt'):
            # Text file processing
            content = str(uploaded_file.read(), "utf-8")
            st.subheader("📝 File Content")
            st.text_area("Content:", content, height=200)
            
            # Basic analysis
            lines = content.split('\n')
            words = content.split()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Lines", len(lines))
            with col2:
                st.metric("Words", len(words))
            with col3:
                st.metric("Characters", len(content))
        
        elif uploaded_file.name.endswith('.csv'):
            # CSV file processing
            df = pd.read_csv(uploaded_file)
            st.subheader("📊 CSV Data")
            st.dataframe(df.head())
            
            st.subheader("📈 Data Info")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Rows**: {len(df)}")
                st.write(f"**Columns**: {len(df.columns)}")
            with col2:
                st.write("**Column Types**:")
                for col, dtype in df.dtypes.items():
                    st.write(f"- {col}: {dtype}")

# System info
st.sidebar.markdown("---")
st.sidebar.subheader("📊 System Status")
st.sidebar.write("✅ Streamlit: Running")
st.sidebar.write("✅ Pandas: Available")
st.sidebar.write("✅ NumPy: Available")
st.sidebar.write("✅ Matplotlib: Available")

# Footer
st.markdown("---")
st.markdown("**Test AI Applications** - Simple working versions")
st.markdown("All features working without AI model dependencies")