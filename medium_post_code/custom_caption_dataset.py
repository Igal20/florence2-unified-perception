import os
import ast
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset

# Constants
DEFAULT_SPLIT_PERCENTAGES = [0.8, 0.1, 0.1]
DEFAULT_TOKENS_DICT = {
    'emotion': '<emo>',
    'pose': '<pose>',
    'jersey_number': '<jnu>',
    'jersey_name': '<jna>',
    'jersey_color': '<jco>',
    'team_name': '<tname>',
    'image_ranking_score': '<ims>',
    'hashtags': '<hashtags>',
    'general_description': '<gdesc>',
    'character_coordinates': '<od>'
}


class BaseDataset(Dataset):
    def __init__(self, split: str):
        self._split = split
        self.name = "BaseDataset"
        self.data = []
        self.task_prompt = ""

    def __len__(self):
        return len(self.data)

    @staticmethod
    def format_text_case_and_punctuation(text: str, is_question: bool = False) -> str:
        if text and text[0].islower():
            text = text.capitalize()
        if not text.endswith(".") and not is_question:
            text += "."
        if not text.endswith("?") and is_question:
            text += "?"
        return text


class CustomCaptionDataset(BaseDataset):
    def __init__(self, split: str, images_folder: Path, jsons_folder: Path,
                 task_prompt: str = None, caption_keys: List[str] = None,
                 max_players: int = 10, convert_to_tokens: bool = True,
                 split_percentages: List[float] = None, manual_seed: int = 42):
        super().__init__(split)
        self.name = "custom_captions"
        self.images_folder = images_folder
        self.jsons_folder = jsons_folder
        self.caption_keys = caption_keys or []
        self.task_prompt = task_prompt
        self.max_players = max_players
        self.convert_to_tokens = convert_to_tokens
        self.tokens_dict = DEFAULT_TOKENS_DICT

        logger.info(f"Note: Only the first {self.max_players} characters will be used if more are detected.")

        self.data = self._load_data(split_percentages or DEFAULT_SPLIT_PERCENTAGES, manual_seed)[split]
        self.parsed_data = self._parse_json_data()

    @staticmethod
    def _get_image_files(folder_path: str, extensions=('.png', '.jpg', '.jpeg')) -> List[str]:
        """Retrieve a list of image files with specified extensions from the folder."""
        return [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(extensions)
        ]

    def _load_data(self, split_percentages: List[float], manual_seed: int) -> Dict[str, List[Tuple[str, str]]]:
        """Load images and JSONs from the folders and split into train, validation, and test sets."""
        if abs(sum(split_percentages) - 1.0) >= 1e-6:
            raise ValueError("Split percentages must sum to 1.0")

        data = []
        image_files = self._get_image_files(str(self.images_folder))
        for img_path in image_files:
            json_path = os.path.join(self.jsons_folder, os.path.splitext(os.path.basename(img_path))[0] + '.json')
            if os.path.exists(json_path):
                data.append((img_path, json_path))
            else:
                logger.warning(f"JSON file does not exist for image: {img_path}")

        random.seed(manual_seed)
        random.shuffle(data)

        train_len = int(len(data) * split_percentages[0])
        val_len = int(len(data) * split_percentages[1])
        return {
            'train': data[:train_len],
            'validation': data[train_len:train_len + val_len],
            'test': data[train_len + val_len:]
        }

    def _parse_json_data(self) -> List[Tuple[str, str]]:
        """Parse JSON data and return annotations as a string."""
        parsed_data = []
        for img_path, json_path in self.data:
            try:
                with open(json_path, "r") as f:
                    json_data = json.load(f)
                annotations = self._filter_annotations(json_data)

                if self.convert_to_tokens:
                    parsed_string = ' '.join(
                        [' '.join([f'{key} {value}' for key, value in d.items()]) for d in annotations]
                    ).replace("<od> ", "").replace("<od>", "")
                else:
                    parsed_string = str(annotations)

                parsed_data.append((img_path, parsed_string))
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.error(f"Error processing JSON file {json_path}: {e}")
        return parsed_data

    def __getitem__(self, idx: int) -> Tuple[str, Image.Image]:
        annotations, image_path = self.parsed_data[idx]
        return annotations, Image.open(image_path)

    @staticmethod
    def _convert_coordinates(coordinates: List[float]) -> str:
        """Convert normalized coordinates to tokens with scaled values."""
        return "".join([f"<loc_{int(coord * 1000)}>" for coord in coordinates])

    def _filter_annotations(self, json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter and process annotations based on caption keys."""
        filtered = []
        characters = json_data.get('characters', [])[:self.max_players]

        for character in characters:
            char_annotations = {}
            for key in self.caption_keys:
                if key in character:
                    if key == 'character_coordinates':
                        char_annotations[self.tokens_dict[key]] = self._convert_coordinates(character[key])
                    else:
                        char_annotations[self.tokens_dict[key]] = character[key]
            filtered.append(char_annotations)

        if 'general_description' in self.caption_keys:
            filtered.append({self.tokens_dict['general_description']: json_data.get('general_description', '')})
        if 'hashtags' in self.caption_keys:
            hashtags = ast.literal_eval(json_data.get('hashtags', '[]'))
            filtered.append({self.tokens_dict['hashtags']: ', '.join(hashtags)})
        if 'image_ranking_score' in self.caption_keys:
            filtered.append(
                {self.tokens_dict['image_ranking_score']: str(int(json_data.get('image_ranking_score', 0)))})

        return filtered


def main():
    images_folder = Path(r"C:\path\to\images")
    jsons_folder = Path(r"C:\path\to\annotations")
    split = "train"
    filter_keys = ["character_coordinates", "emotion", "pose", "jersey_color", "jersey_number", "jersey_name",
                   "general_description"]

    dataset = CustomCaptionDataset(
        split=split,
        images_folder=images_folder,
        jsons_folder=jsons_folder,
        caption_keys=filter_keys,
        convert_to_tokens=True
    )

    print(dataset[0])
    print(f"Dataset size: {len(dataset)}")


if __name__ == '__main__':
    main()
