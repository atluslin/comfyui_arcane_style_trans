import os
import torch
from facenet_pytorch import MTCNN
from torchvision import transforms
from PIL import Image
import numpy as np

class ArcaneGANNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # ComfyUI's image input type
                "width": ("INT", {
                    "default": 512,    # 默认宽度
                    "min": 256,        # 最小宽度
                    "max": 4096,       # 最大宽度
                    "step": 8,         # 步长，确保为偶数
                    "display": "number"
                }),
                "height": ("INT", {
                    "default": 512,    # 默认高度
                    "min": 256,        # 最小高度
                    "max": 4096,       # 最大高度
                    "step": 8,         # 步长，确保为偶数
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)  # ComfyUI's image output type
    FUNCTION = "process_image"
    CATEGORY = "🔥🔥🔥ljeasynode🔥🔥🔥/风格化"

    def __init__(self):
        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(image_size=256, margin=80)
        
        # Load the ArcaneGAN v0.4 model from the local 'models' folder
        model_path = os.path.join(os.path.dirname(__file__), "models", "ArcaneGANv0.4.jit")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please place ArcaneGANv0.4.jit in the 'models' folder.")
        
        self.model = torch.jit.load(model_path).eval().cuda().half()

        # Define normalization parameters
        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.229, 0.224, 0.225]
        self.t_stds = torch.tensor(self.stds).cuda().half()[:, None, None]
        self.t_means = torch.tensor(self.means).cuda().half()[:, None, None]

        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.means, self.stds)
        ])

    def detect(self, img):
        """Detect faces using MTCNN."""
        batch_boxes, batch_probs, batch_points = self.mtcnn.detect(img, landmarks=True)
        if not self.mtcnn.keep_all:
            batch_boxes, batch_probs, batch_points = self.mtcnn.select_boxes(
                batch_boxes, batch_probs, batch_points, img, method=self.mtcnn.selection_method
            )
        return batch_boxes, batch_points

    def makeEven(self, _x):
        """Ensure dimensions are even."""
        return int(_x) if (_x % 2 == 0) else int(_x + 1)

    def scale_to_resolution(self, _img, target_width, target_height, target_face=256, max_upscale=2):
        """Scale the image to the specified resolution while considering face size."""
        boxes, _ = self.detect(_img)
        x, y = _img.size
        ratio = 2  # Initial ratio

        # Scale based on face size if detected
        if boxes is not None and len(boxes) > 0:
            face_size = max(boxes[0][2:] - boxes[0][:2])
            ratio = target_face / face_size
            ratio = min(ratio, max_upscale)

        # Calculate intermediate dimensions
        x_scaled = x * ratio
        y_scaled = y * ratio

        # Adjust to target resolution while preserving aspect ratio
        target_ratio = target_width / target_height
        current_ratio = x_scaled / y_scaled

        if current_ratio > target_ratio:
            # Image is wider than target, fit to width
            final_width = target_width
            final_height = int(final_width / current_ratio)
        else:
            # Image is taller than target, fit to height
            final_height = target_height
            final_width = int(final_height * current_ratio)

        # Ensure even dimensions
        final_width = self.makeEven(final_width)
        final_height = self.makeEven(final_height)

        # Resize to final dimensions
        return _img.resize((final_width, final_height))

    def tensor2im(self, var):
        """Convert tensor back to image."""
        return var.mul(self.t_stds).add(self.t_means).mul(255.).clamp(0, 255).permute(1, 2, 0)

    def proc_pil_img(self, input_image):
        """Process the image with ArcaneGAN v0.4."""
        transformed_image = self.img_transforms(input_image)[None, ...].cuda().half()
        with torch.no_grad():
            result_image = self.model(transformed_image)[0]
            output_image = self.tensor2im(result_image)
            output_image = output_image.detach().cpu().numpy().astype('uint8')
            return Image.fromarray(output_image)

    def process_image(self, image, width, height):
        """Main processing function for ComfyUI."""
        # Convert ComfyUI image (tensor) to PIL Image
        if isinstance(image, torch.Tensor):
            image = image.squeeze(0)  # Remove batch dimension if present
            image = (image * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
            pil_image = Image.fromarray(image)

        # Scale to the specified resolution
        scaled_image = self.scale_to_resolution(pil_image, target_width=width, target_height=height, target_face=256, max_upscale=1)
        
        # Process with ArcaneGAN
        output_pil = self.proc_pil_img(scaled_image)

        # Resize to exact target resolution if needed
        if output_pil.size != (width, height):
            output_pil = output_pil.resize((width, height), Image.LANCZOS)

        # Convert back to ComfyUI tensor format [B, H, W, C], range [0, 1]
        output_tensor = torch.from_numpy(np.array(output_pil)).float() / 255.0
        output_tensor = output_tensor.unsqueeze(0)  # Add batch dimension
        return (output_tensor,)

# Register the node with ComfyUI
NODE_CLASS_MAPPINGS = {
    "Arcane_style_trans": ArcaneGANNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Arcane_style_trans": "Arcane_style_trans"
}

DEFAULT_CATEGORY = '🔥🔥🔥ljeasynode🔥🔥🔥'