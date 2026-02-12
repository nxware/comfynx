
import os
import io
from PIL import Image,  ImageDraw, ImageFont
from typing import Optional, Tuple

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

    def create_text_image(
        self, 
        text: str,
        color: Tuple[int, int, int] | Tuple[int, int, int, int] = (255, 255, 255),
    ) -> Image.Image:
        """
        Create a transparent RGBA image containing the given text.

        :param text: Text to render (supports multiline with \n)
        :param color: (R, G, B) or (R, G, B, A)
        :return: RGBA PIL Image
        """

        if not text:
            raise ValueError("Text must not be empty")

        if len(color) == 3:
            r, g, b = color
            a = 255
        elif len(color) == 4:
            r, g, b, a = color
        else:
            raise ValueError("Color must be an (R, G, B) or (R, G, B, A) tuple")

        # Basic range validation
        for channel in (r, g, b, a):
            if not (0 <= channel <= 255):
                raise ValueError("Color channel values must be in range 0–255")

        font = ImageFont.load_default()

        # Measure text
        dummy = Image.new("RGBA", (1, 1))
        draw = ImageDraw.Draw(dummy)
        bbox = draw.multiline_textbbox((0, 0), text, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        padding = 10

        # Transparent background
        img = Image.new("RGBA", (width + padding * 2, height + padding * 2), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        draw.multiline_text(
            (padding, padding),
            text,
            fill=(r, g, b, a),
            font=font,
        )

        return img

    def add_watermark(self, image: Image.Image, watermark_file: str, margin_x: int = 10, margin_y: int = 10) -> Image.Image:
        if not os.path.exists(watermark_file):
            raise FileNotFoundError(f"Watermark not found: {watermark_file}")
        watermark = Image.open(watermark_file).convert("RGBA")
        return self.add_watermark_from_image(image, watermark, margin_x, margin_y)

    def add_watermark_from_image(self, image, watermark: Image.Image, margin_x: int = 10, margin_y: int = 10) -> Image.Image:
        base = image.convert("RGBA")
        base_w, base_h = base.size
        wm_w, wm_h = watermark.size
        position = (base_w - wm_w - margin_x, base_h - wm_h - margin_y)
        result = base.copy()
        result.alpha_composite(watermark, position)
        return result
    
    def image_to_bytesio(
        self,
        img: Image.Image,
        format: str = "PNG",
        **save_kwargs,
    ) -> io.BytesIO:
        """
        Serialize a PIL Image into a BytesIO buffer.

        :param img: PIL Image instance
        :param format: Output format (e.g., "PNG", "JPEG")
        :param save_kwargs: Additional Pillow save() parameters
        :return: BytesIO positioned at start
        """
        if not isinstance(img, Image.Image):
            raise TypeError("img must be a PIL.Image.Image instance")
        format = format.upper()
        if format == "JPG":
            format = "JPEG"
        output_img = img
        if format == "JPEG":
            output_img = img.convert("RGB")
        buffer = io.BytesIO()
        output_img.save(buffer, format=format, **save_kwargs)
        buffer.seek(0)
        return buffer

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