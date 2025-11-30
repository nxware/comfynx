import os
from PIL import Image
from typing import Optional

# ComfyUI custom node: AddWatermark
# Place this file in ComfyUI/custom_nodes/
# Restart ComfyUI afterward.

class AddWatermark:
    """
        Adds a transparent PNG watermark to the bottom-right corner of an input image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            "image": ("IMAGE",),
            "watermark_path": ("STRING", {"default": "watermark.png"}),
            "margin_x": ("INT", {"default": 10, "min": 0, "max": 200}),
            "margin_y": ("INT", {"default": 10, "min": 0, "max": 200}),
        }
    }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "image/postprocessing"

    def apply(self, image, watermark_path: str, margin_x: int, margin_y: int):
        pil_image = self.tensor_to_pil(image)
        result = self.add_watermark(pil_image, watermark_path, margin_x, margin_y)
        return (self.pil_to_tensor(result),)

    def add_watermark(self, image: Image.Image, watermark_file: str, margin_x: int, margin_y: int) -> Image.Image:
        if not os.path.exists(watermark_file):
            raise FileNotFoundError(f"Watermark not found: {watermark_file}")

        base = image.convert("RGBA")
        watermark = Image.open(watermark_file).convert("RGBA")

        base_w, base_h = base.size
        wm_w, wm_h = watermark.size

        position = (base_w - wm_w - margin_x, base_h - wm_h - margin_y)

        result = base.copy()
        result.alpha_composite(watermark, position)

        return result

    def tensor_to_pil(self, tensor_image):
        if isinstance(tensor_image, Image.Image):  # If already a PIL image, pass through
            return tensor_image
        # Otherwise convert ComfyUI tensor image: [batch, height, width, channels]
        import numpy as np
        img_np = 255. * tensor_image[0].cpu().numpy()
        print(str(img_np.shape))
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)
        return pil_image

    def pil_to_tensor(self, pil_image: Image.Image):
        import torch
        import numpy as np
        img_np = np.array(pil_image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np)
        return torch.stack([img_tensor])


class SaveJpegImage:
    def __init__(self):
        import folder_paths
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 80

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."}),
                "filename_postfix": ("STRING", {"default": "_"}),
                "enable": ("BOOLEAN", {"default": True})
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."

    def save_images(self, images, filename_prefix="ComfyUI", filename_postfix="", enable=True, prompt=None, extra_pnginfo=None):
        import folder_paths
        import numpy as np
        if not enable:
            return {}
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)).convert("RGB")
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}{filename_postfix}.jpg"
            img.save(os.path.join(full_output_folder, file), compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }


class ImageQueueLoader:
    """
     Lädt ein Image aus einem Verzeichnis und löscht das Image dann,
     für Batch und i2i Workflows
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            "image": ("IMAGE",),
            "path": ("STRING", {"default": "."}),
            }
        }


class StringMultiConcat:
    """
     
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
              "s1": ("STRING", {"default": ""}),
              "s2": ("STRING", {"default": ""}),
              "s3": ("STRING", {"default": ""}),
              "s4": ("STRING", {"default": ""}),
              "s5": ("STRING", {"default": ""}),
              "s6": ("STRING", {"default": ""}),
              "s7": ("STRING", {"default": ""}),
              "s8": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "concat"

    OUTPUT_NODE = True

    CATEGORY = "util"
    DESCRIPTION = "Concat."

    def concat(self, s1, s2, s3, s4, s5, s6, s7, s8):
        if not isinstance(s1, str):
            s1 = s1[0]
        if not isinstance(s2, str):
            s2 = s2[0]
        if not isinstance(s3, str):
            s3 = s3[0]
        if not isinstance(s4, str):
            s4 = s4[0]
        if not isinstance(s5, str):
            s5 = s5[0]
        if not isinstance(s6, str):
            s6 = s6[0]
        if not isinstance(s7, str):
            s7 = s7[0]
        if not isinstance(s8, str):
            s8 = s8[0]
        res = str(s1)+str(s2)+str(s3)+str(s4)+str(s5)+str(s6)+str(s7)+str(s8)
        print("CONCAT: " + res)
        return (res, )
    

class JsonArrayPicker:
    """
     
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
              "json_str": ("STRING", {"default": "[]"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_str"

    OUTPUT_NODE = True

    IS_CHANGED = True
    CATEGORY = "util"
    DESCRIPTION = "Concat."

    def generate_str(self, json_str):
        import json
        import random
        a = json.loads(json_str)
        return (random.choice(a), )


class FileLinePicker:
    """
     
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
              "path": ("STRING", {"default": "./prompts.txt"})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_str"

    IS_CHANGED = True
    CATEGORY = "util"
    DESCRIPTION = str(__doc__)

    def generate_str(self, path):
        import random
        with open(path, "r") as f:
            line = random.choice(f.readlines())
            return (line, )



class StringMultilineC:
    """
     MultilineString with Comments #
    """
    CATEGORY = "utils/primitive"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
              "text": ("STRING", {"default": "", "multiline":True})
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_str"
    DESCRIPTION = str(__doc__)

    def generate_str(self, text):
        if not isinstance(text, str):
            text = text[0]
        res = "".join(filter(lambda s: not s.startswith("#"), text.splitlines()))
        return (res,)


class Translate:
    """
    """
    CATEGORY = "utils/primitive"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
              "path": ("STRING", {"default": "translate.json"}),
              "prompt": ("STRING", {"default": ""})
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_str"
    DESCRIPTION = str(__doc__)

    def generate_str(self, path, prompt):
        if not isinstance(path, str):
            path = path[0]
        if not isinstance(path, str):
            prompt = prompt[0]
        import json
        with open(path, 'r') as f:
            d = json.load(f)
        for key in d.keys():
            prompt = prompt.replace(key, d[key])
        return (prompt,)


def init():
    from .comfynx import web
    web.setup_routes()


NODE_CLASS_MAPPINGS = {
    "AddWatermark": AddWatermark,
    "SaveJpegImage": SaveJpegImage,
    "StringMultiConcat": StringMultiConcat,
    "JsonArrayPicker": JsonArrayPicker,
    "FileLinePicker": FileLinePicker,
    "StringMultilineC": StringMultilineC,
    "Translate": Translate
}