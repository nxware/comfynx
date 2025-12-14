
import os
import time
from PIL import Image

from nwebclient import NWebClient

class NWebUpload:

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
                "nweb_name": ("STRING", {"default": "default"}),
                "group": ("STRING", {"default": "incoming"}),
                "enable": ("BOOLEAN", {"default": True}),
                "keepfile": ("BOOLEAN", {"default": False})
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


    def save_images(self, images, filename_prefix="ComfyUI", filename_postfix="", nweb_name="default", group="", enable=True, keepfile=False, prompt=None, extra_pnginfo=None):
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
            f_path = os.path.join(full_output_folder, file)
            img.save(f_path, compress_level=self.compress_level)

            nc = NWebClient(nweb_name)
            with open(f_path, 'rb') as f:
                doc = nc.createFileDoc(file, group, f)
                doc.setMetaValue('ai', 'prompt', str(prompt))
            if keepfile:
                results.append({
                    "filename": file,
                    "subfolder": subfolder,
                    "type": self.type
                })
            else:
                time.sleep(0.3)
                os.remove(f_path)
            counter += 1

        return { "ui": { "images": results } }
    

class NWebDocMetaUpdate:
    def __init__(self):
        self.ns = 'ai'    

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "nweb_name": ("STRING", {"default": "default"}),
                "ns": ("STRING", {"default": "ai"}),
                "name": ("STRING", {"default": "suggested_tags"}),
                "doc": ("STRING", {"default": "1234"}),
                "value": ("STRING", {"default": "1234"}),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "update_doc"

    OUTPUT_NODE = True

    CATEGORY = "image"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."

    def update_doc(self, nweb_name, ns, name, doc, value):
        nc = NWebClient(nweb_name)
        d = nc.doc(doc)
        d.setMetaValue(ns, name, value)
        return {}
    

NODE_CLASS_MAPPINGS = {
  "NWebUpload": NWebUpload,
  "NWebDocMetaUpdate": NWebDocMetaUpdate,
}