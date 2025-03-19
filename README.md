# AI-Enabled Choreography: Dance Beyond Music

## Introduction
This is my submission for GSoC AI-Enabled Choreography. In this project, I fuse art with deep learning by:
- Visualizing 3D dance sequences with BPM-based animation.
- Training a multimodal model that aligns dance (motion capture) sequences with descriptive text using contrastive learning.

I explore how technology can capture the fluidity of dance and the expressiveness of language, making it easier to generate and understand artistic movement in sync with music.

---

## Part 1: Animate some dance data

### Overview
The animation module visualizes 3D dance sequences with a speed that adapts based on the audio's BPM. This allows me to compare the original sequence with any generated sequence effectively.

### Key Features
- **BPM Extraction:**  
  The `extract_bpm` function loads an audio file and extracts its beats per minute (BPM) using the `librosa` library. It then computes a speed multiplier relative to a baseline BPM to synchronize the animation.

- **3D Animation:**  
  The `animate_dance_sequence` function takes a NumPy array of joint positions, optionally trims the sequence, and animates it in a 3D space. I also support visualizing a skeleton by connecting joints, which makes it easier to see differences between sequences.

---

## Part2: Train a multimodal model of dance & text

### Model Selection
I opted for a VAE structure with LSTM layers for this model. LSTM is well-suited for capturing the sequential nature of dance data, whereas GNNs, despite modeling joint relationships, are computationally more demanding and less effective for our generation task.

### Data Processing
- **Preprocessing:** Center the motion capture data.
- **Dataset Split:**  
  - 70% Training  
  - 15% Validation  
  - 15% Testing
- **Sequence Length:** Set to 64 (with no overlap).

### Model Design
- **Encoders:**  
  - **Dance Encoder (`DacneEncoder`):**  
    Utilizes a 2-layer bidirectional LSTM, applies mean pooling over time, and projects the result to an embedding space via a linear layer. *(Ensure the initializer is named `__init__`.)*
  - **Text Encoder (`TextEncoder`):**  
    Uses an embedding layer followed by a GRU to process text, with mean pooling and a linear projection to the same embedding space. *(Rename the method from `forwad` to `forward`.)*

- **Loss Function:**  
  Implements a contrastive loss (InfoNCE) by comparing dance and text embeddings (scaled by a temperature parameter) using cross-entropy loss in both directions.

- **Training & Generation:**  
  - **Training:** Uses the Adam optimizer to update model parameters over multiple epochs.  
  - **Generation:**  
    - `generate_dance_from_text`: Retrieves the dance sequence that best matches a text input.  
    - `generate_text_from_dance`: Retrieves the text description that best corresponds to a dance input.
