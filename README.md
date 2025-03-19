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

The model consists of two main components:

1. **DanceEncoder**: An LSTM-based encoder that processes motion capture sequences
   - Input: Dance sequence of shape (sequence_length, num_joints*3)
   - Output: Normalized embedding vector

2. **TextEncoder**: A GRU-based encoder that processes text descriptions
   - Input: Tokenized text sequence
   - Output: Normalized embedding vector

The model is trained using contrastive learning (InfoNCE loss) to align dance and text embeddings in the same space.

## Results

![This plot shows the training and validation loss over epochs. The training loss (blue line) decreases from approximately 0.8 to 0.1, indicating that the model is learning to align dance and text embeddings in the shared space. The validation loss (red line) follows a similar trend but is slightly higher, which is expected and indicates that the model is generalizing well without significant overfitting.]
