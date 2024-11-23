import json
from PIL import Image
from torchvision import transforms
from random import choice
import torch
from torch.utils.data import Dataset
import os
import random
from transformers import CLIPImageProcessor

class TryonDataset(Dataset):
    def __init__(self, json_file, tokenizer, size, image_root_path):
        
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"{json_file} not exists")

        if isinstance(json_file, str):
            with open(json_file, "r") as file:
                self.data = json.load(file)
        elif isinstance(json_file, list):
            for file_path in json_file:
                with open(file_path, "r") as file:
                    data = json.load(file)
                    if not hasattr(self, "data"):
                        self.data = data
                    else:
                        self.data.extent(data)
        else:
            raise ValueError(f"Input json_file should either be a Str or List")

        self.tokenizer = tokenizer
        self.size = size
        self.image_root_path = image_root_path
        self.clip_image_processor = CLIPImageProcessor()
        self.transform = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop([640, 512]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        person_path = item["image_file"]
        cloth_path = item["cloth_file"]
        person_img = Image.open(person_path).convert("RGB")
        cloth_img  = Image.open(cloth_path).convert("RGB")
        text = choice(item["text"])

        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < 0.05:
            drop_image_embed = 1
        elif rand_num < 0.1:
            text = ""
        elif rand_num < 0.15:
            text = ""
            drop_image_embed = 1
        
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        null_text_input_ids = self.tokenizer(
            "",
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        vae_person = self.transform(person_img)
        vae_clothes = self.transform(cloth_img)
        clip_image =  self.clip_image_processor(images=cloth_img, return_tensors="pt").pixel_values

        return {
            "vae_person": vae_person,
            "vae_clothes" : vae_clothes,
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed,
            "text": text,
            "text_input_ids": text_input_ids,
            "null_text_input_ids": null_text_input_ids
        }
    
def collate_fn(batch_data):
    vae_person = torch.stack([example["vae_person"] for example in batch_data]).to(memory_format = torch.contiguous_format).float()
    vae_clothes = torch.stack([example["vae_clothes"] for example in batch_data]).to(memory_format=torch.contiguous_format).float()
    clip_image = torch.cat([example["clip_image"] for example in batch_data], dim=0)
    drop_image_embed = [example["drop_image_embed"] for example in batch_data]

    text = [example["text"] for example in batch_data]
    input_ids = torch.cat([example["text_input_ids"] for example in batch_data], dim=0)
    null_input_ids = torch.cat([example["null_text_input_ids"] for example in batch_data], dim=0)

    return {
        "vae_person": vae_person,
        "vae_clothes": vae_clothes,
        "clip_image": clip_image,
        "drop_image_embed": drop_image_embed,
        "text": text,
        "input_ids": input_ids,
        "null_input_ids": null_input_ids,
    }