# Expanding Florence-2's Vocabulary: A Guide to Adding Custom Tokens During Fine-Tuning

## Overview
This repository contains an **updated version** of the `processing_florence2.py` file, which extends the **Florence-2 vision-language model (VLM)** with **custom tokens** and **expanded vocabulary** for fine-tuning. These modifications enhance the model’s ability to understand specialized image attributes in a **single inference pass**, reducing redundancy and improving accuracy.

Detailed explanations of this modification process can be found in the **Medium blog post:**  
🔗 [Expanding Florence-2's Vocabulary: A Guide to Adding Custom Tokens During Fine-Tuning](#)  *(Insert actual link here)*

## Why Expand Florence-2’s Vocabulary?
While Florence-2 is a powerful open-source multimodal model, it may struggle with **domain-specific terminology** or **visual elements** unique to specialized applications. This update is particularly useful for:
- **Sports Analysis** – Extracting player attributes such as **jersey number, pose, emotion, and team name**.
- **Medical Image Processing** – Identifying anatomical structures and medical conditions.
- **E-Commerce Tagging** – Enhancing product descriptions with structured metadata.

## Key Enhancements in `processing_florence2.py`

1. **Custom Token Integration**
   - Introduces domain-specific tokens such as `<emo>`, `<pose>`, `<team>`, and `<color>`.
   - Allows the model to generate structured outputs containing these tokens.

2. **Tokenizer Update**
   - Modifies `tokenizer.json` to recognize and process new tokens.
   - Ensures token embeddings align with Florence-2’s existing vocabulary.

3. **New Task Definition**
   - Adds a new pipeline task to extract structured **player attributes**.
   - Maps input prompts to the appropriate Florence-2 processing functions.

4. **Enhanced Post-Processing**
   - Implements output formatting to improve structured data retrieval.
   - Ensures bounding box and metadata alignment for downstream applications.


## Contribution
Feel free to open an **issue** or **pull request** if you find any bugs or want to suggest improvements!

---

For a **detailed explanation**, refer to the Medium article:  
🔗 [Expanding Florence-2's Vocabulary: A Guide to Adding Custom Tokens During Fine-Tuning](#) *(Insert actual link here)*

