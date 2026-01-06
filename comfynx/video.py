
import torch

class ConcatImageBatches:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images_a": ("IMAGE",),
                "images_b": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "concat_batches"
    CATEGORY = "image/batch"

    def concat_batches(self, images_a, images_b):
        # Erwartetes Format: [B, H, W, C]

        if images_a.shape[1:] != images_b.shape[1:]:
            raise ValueError(
                "Images müssen identische H, W und C Dimensionen haben"
            )

        result = torch.cat((images_a, images_b), dim=0)
        return (result,)


class AnimateImageSlidingCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "result_width": ("INT", {"default": 512, "min": 1}),
                "result_height": ("INT", {"default": 512, "min": 1}),
                "mode": (["Horizontal", "Vertical", "DiagonalUp", "DiagonalDown"],),
                "step_size": ("INT", {"default": 1, "min": 1}),
                "frames": ("INT", {"default": 0, "min": 0}),  # 0 = auto
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "animate"
    CATEGORY = "image/animation"

    def animate(self, image, result_width, result_height, mode, step_size, frames):
        if image.shape[0] != 1:
            raise ValueError("Input image muss Batch-Größe 1 haben")

        _, img_h, img_w, channels = image.shape

        if img_w < result_width or img_h < result_height:
            raise ValueError(
                "Input image muss mindestens so groß sein wie result_width und result_height"
            )

        max_x = img_w - result_width
        max_y = img_h - result_height

        # maximale Frames basierend auf Mode und step_size
        if mode == "Horizontal":
            max_frames = (max_x // step_size) + 1
        elif mode == "Vertical":
            max_frames = (max_y // step_size) + 1
        else:  # DiagonalUp / DiagonalDown
            max_frames = (min(max_x, max_y) // step_size) + 1

        if frames > 0:
            total_frames = min(frames, max_frames)
        else:
            total_frames = max_frames

        device = image.device
        dtype = image.dtype

        output = torch.empty(
            (total_frames, result_height, result_width, channels),
            device=device,
            dtype=dtype
        )

        for i in range(total_frames):
            shift = i * step_size

            if mode == "Horizontal":
                x = shift
                y = (img_h - result_height) // 2
            elif mode == "Vertical":
                x = (img_w - result_width) // 2
                y = shift
            elif mode == "DiagonalDown":
                x = shift
                y = shift
            elif mode == "DiagonalUp":
                x = shift
                y = max_y - shift

            # Safety Clamp
            x = max(0, min(x, max_x))
            y = max(0, min(y, max_y))

            output[i] = image[
                0,
                y:y + result_height,
                x:x + result_width,
                :
            ]

        return (output,)



class ImageToBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "n_frames": ("INT", {"default": 8, "min": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "to_batch"
    CATEGORY = "image/batch"

    def to_batch(self, image, n_frames):
        # image: [1, H, W, C]
        if image.shape[0] != 1:
            raise ValueError("Input image muss Batch-Größe 1 haben")

        batch = image.repeat(n_frames, 1, 1, 1)  # [n_frames, H, W, C]
        return (batch,)


class LastImageFromBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_batch": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "get_last"
    CATEGORY = "image/batch"

    def get_last(self, image_batch):
        # image_batch: [B, H, W, C]
        if image_batch.shape[0] < 1:
            raise ValueError("Batch ist leer")

        last_image = image_batch[-1:]  # behält die Batch-Dimension [1, H, W, C]
        return (last_image,)


NODE_CLASS_MAPPINGS = {
    "AnimateImageSlidingCrop": AnimateImageSlidingCrop,
    "ConcatImageBatches": ConcatImageBatches,
    "ImageToBatch": ImageToBatch,
    "LastImageFromBatch": LastImageFromBatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimateImageSlidingCrop": "Animate Image Sliding Crop",
    "ConcatImageBatches": "Concat Image Batches",
    "ImageToBatch": "Image to Batch",
    "LastImageFromBatch": "Last Image from Batch"
}