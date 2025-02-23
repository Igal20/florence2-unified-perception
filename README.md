# Expanding Florence-2's Vocabulary: An Advanced Guide to Adding Custom Tokens During Fine-Tuning
## Overview
This repository contains an **updated version** of the `processing_florence2.py` file, which extends the **Florence-2 vision-language model (VLM)** with **custom tokens** and **expanded vocabulary** for fine-tuning. These modifications enhance the model’s ability to understand specialized image attributes in a **single inference pass**, reducing redundancy and improving accuracy.

Detailed explanations of this modification process can be found in the **Medium blog post:**  
🔗 [Expanding Florence-2's Vocabulary: A Guide to Adding Custom Tokens During Fine-Tuning](https://medium.com/@ygal20/expanding-florence-2s-vocabulary-an-advanced-guide-to-adding-custom-tokens-during-fine-tuning-138fab660b64)


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


## Explanation of `custom_caption_dataset.py`

The `custom_caption_dataset.py` file defines a PyTorch dataset class that processes images and their corresponding JSON annotations for training and fine-tuning Florence-2 with structured captions. It ensures proper dataset preparation by loading, filtering, and formatting annotations into a structured format.

### Key Features
1. **Dataset Splitting** – The dataset is split into training, validation, and test sets based on predefined percentages.
2. **Tokenized Captions** – Converts raw annotations into structured text with predefined tokens (e.g., `<emo>` for emotion, `<pose>` for pose).
3. **Image-Annotation Pairing** – Matches each image with its corresponding annotation JSON file.
4. **Annotation Parsing** – Extracts and formats relevant attributes such as character coordinates, team names, and general descriptions.
5. **Configurable Settings** – Allows customization of caption keys, number of characters processed, and whether to tokenize outputs.

### How It Works
- The dataset scans a folder containing images and their corresponding JSON annotations.
- It reads and filters relevant attributes from the JSON files.
- It converts the attributes into a structured text format, replacing raw values with predefined tokens.
- It loads image-caption pairs into a PyTorch dataset for use in training models like Florence-2.

### Example Usage
```python
from pathlib import Path
from custom_caption_dataset import CustomCaptionDataset

images_folder = Path("/path/to/images")
jsons_folder = Path("/path/to/annotations")
split = "train"
caption_keys = ["character_coordinates", "emotion", "pose", "jersey_color", "jersey_number", "jersey_name", "general_description"]

dataset = CustomCaptionDataset(
    split=split,
    images_folder=images_folder,
    jsons_folder=jsons_folder,
    caption_keys=caption_keys,
    convert_to_tokens=True
)

print(dataset[0])
print(f"Dataset size: {len(dataset)}")
```


## Example of an Annotation File

The `example_of_annotations_file.json` provides a structured format for image metadata, including details about the scene, characters, and their attributes.

### JSON Structure
```json
{
    "general_description": "The image depicts an intense National Hockey League (NHL) game between two teams in the 2023-24 season. Players are actively engaged in the match. The scene is filled with action, with players focusing on controlling the puck on the ice.",
    "image_size": [
        1080,
        1920
    ],
    "number_of_characters": 4,
    "hashtags": "['#NHL', '#Hockey', '#GameDay', '#IceHockey', '#Sports']",
    "image_ranking_score": 8.0,
    "original_image_url": "URL_TO_AN_IMAGE",
    "image_id": 9999,
    "characters": [
        {
            "character": 4,
            "character_coordinates": [
                0.721,
                0.015,
                0.945,
                0.632
            ],
            "emotion": "Focused",
            "pose": "In Action",
            "jersey_color": "White and Green",
            "jersey_number": "None",
            "jersey_name": "None",
            "team_name": "None",
            "is_player": "YES"
        },
        {
            "character": 1,
            "character_coordinates": [
                0.386,
                0.079,
                0.868,
                0.996
            ],
            "emotion": "Focused",
            "pose": "In Action",
            "jersey_color": "White and Green",
            "jersey_number": "92",
            "jersey_name": "None",
            "team_name": "None",
            "is_player": "YES"
        },
        {
            "character": 2,
            "character_coordinates": [
                0.131,
                0.044,
                0.271,
                0.798
            ],
            "emotion": "Focused",
            "pose": "In Action",
            "jersey_color": "Black and Orange",
            "jersey_number": "None",
            "jersey_name": "None",
            "team_name": "None",
            "is_player": "YES"
        },
        {
            "character": 3,
            "character_coordinates": [
                0.209,
                0.131,
                0.542,
                0.996
            ],
            "emotion": "Focused",
            "pose": "In Action",
            "jersey_color": "Black",
            "jersey_number": "19",
            "jersey_name": "None",
            "team_name": "None",
            "is_player": "YES"
        }
    ]
}
```

This JSON file serves as input for `custom_caption_dataset.py`, ensuring structured metadata extraction and enabling Florence-2 to process and understand domain-specific elements effectively.


## Contribution
Feel free to open an **issue** or **pull request** if you find any bugs or want to suggest improvements!

---

For a **detailed explanation**, refer to the Medium article:  
🔗 [Expanding Florence-2's Vocabulary: A Guide to Adding Custom Tokens During Fine-Tuning](#) *(Insert actual link here)*
