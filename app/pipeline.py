
import io
import torch
from PIL import Image
from torchvision import transforms
import sys, os

# Ensure our bundled src/ is importable
CURR_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(CURR_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from stain_transfer import StainTransfer  # provided by user code
from my_utils.training_utils import build_transform

class StainPipeline:
    def __init__(self, pretrained_name=None, pretrained_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize model (either HF name or local path)
        self.model = StainTransfer(pretrained_name=pretrained_name, pretrained_path=pretrained_path)
        self.model.eval()
        try:
            # Not all builds have xformers; it's okay if this fails
            self.model.unet.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        self.model.to(self.device)

        # default transform similar to user's inference script
        self.transform = build_transform("resize_512x512")

    @torch.inference_mode()
    def run(
        self,
        image: Image.Image,
        direction: str | None = None,
        prompt: str | None = None,
    ) -> Image.Image:
        # Convert image to tensor expected by model
        # print(image.shape)
        img = self.transform(image)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.5], [0.5])(img).unsqueeze(0).cuda()
        seed = 42
        if seed is not None:
            torch.manual_seed(seed)
            if self.device.startswith("cuda"):
                torch.cuda.manual_seed_all(seed)

        # The user's model supports two usage modes:
        # - Pretrained model_name: does not require prompt/direction
        # - Custom weights (pretrained_path): requires prompt and direction
        # We'll pass through when provided.
        out = self.model(img, direction=direction, caption=prompt)
        # Convert back to PIL
        if isinstance(out, torch.Tensor):
            out = out.detach().clamp(-1, 1)
            out = (out.add(1).div(2).mul(255).byte())
            out = out.permute(0, 2, 3, 1).cpu().numpy()
            from PIL import Image
            return Image.fromarray(out[0])
        elif isinstance(out, Image.Image):
            return out
        else:
            # Fallback: try to detect a tensor-like
            try:
                import numpy as np
                arr = out[0] if isinstance(out, (list, tuple)) else out
                if hasattr(arr, "cpu"):
                    arr = arr.detach().cpu().numpy()
                return Image.fromarray(arr)
            except Exception as e:
                raise RuntimeError(f"Unexpected model output type: {type(out)}") from e
