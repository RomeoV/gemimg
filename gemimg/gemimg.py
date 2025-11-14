import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Union

import httpx
from dotenv import load_dotenv
from PIL import Image

from .utils import _validate_aspect, b64_to_img, img_b64_part, img_to_b64, save_image

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class GemImg:
    api_key: str = field(default=None, repr=False)
    client: httpx.Client = field(default_factory=httpx.Client, repr=False)
    model: str = "gemini-2.5-flash-image"

    def __post_init__(self):
        # Get API key from environment if not provided
        if self.api_key is None:
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY environment variable is required. "
                    "Set it in .env file or pass api_key parameter."
                )

    def generate(
        self,
        prompt: Optional[str] = None,
        imgs: Optional[Union[str, Image.Image, List[str], List[Image.Image]]] = None,
        aspect_ratio: str = "1:1",
        resize_inputs: bool = True,
        save: bool = True,
        save_dir: str = "",
        temperature: float = 1.0,
        webp: bool = False,
        n: int = 1,
        store_prompt: bool = False,
    ) -> Optional["ImageGen"]:
        if not prompt and not imgs:
            raise ValueError("Either 'prompt' or 'imgs' must be provided")

        if n > 1:
            if temperature == 0:
                raise ValueError(
                    "Generating multiple images at temperature = 0.0 is redundant."
                )
            # Exclude 'self' from locals to avoid conflicts when passing as kwargs
            kwargs = {k: v for k, v in locals().items() if k != "self"}
            return self._generate_multiple(**kwargs)

        query_params, headers, api_url = self._build_request(
            prompt, imgs, resize_inputs, temperature, aspect_ratio
        )

        try:
            response = self.client.post(
                api_url, json=query_params, headers=headers, timeout=120
            )
        except httpx.exceptions.Timeout:
            logger.error("Request Timeout")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred: {e}")
            return None

        response_data = response.json()

        return self._parse_response(
            response_data, save, save_dir, webp, store_prompt, prompt
        )

    def _parse_response(self, response_data, save, save_dir, webp, store_prompt, prompt):
        """Parse response from API (OpenAI-compatible format)."""
        usage_metadata = response_data.get("usage", {})

        choices = response_data.get("choices", [])
        if not choices:
            logger.error("No choices in API response.")
            logger.error(f"Full response: {response_data}")
            return None

        message = choices[0].get("message", {})

        output_images = []

        # Images are returned in a separate 'images' field
        images_data = message.get("images", [])

        if not images_data:
            logger.error("No images field in API response.")
            logger.error(f"Full message: {message}")
            return None

        for item in images_data:
            if isinstance(item, dict):
                if item.get("type") == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    # Extract base64 from data URL
                    if image_url.startswith("data:image"):
                        b64_data = image_url.split(",", 1)[1] if "," in image_url else image_url
                        output_images.append(b64_to_img(b64_data))

        if not output_images:
            logger.error("No images found in API response.")
            return None

        output_image_paths = []
        if save:
            response_id = response_data.get("id", "output")
            file_extension = "webp" if webp else "png"
            for idx, img in enumerate(output_images):
                image_path = (
                    f"{response_id}.{file_extension}"
                    if len(output_images) == 1
                    else f"{response_id}-{idx}.{file_extension}"
                )
                full_path = os.path.join(save_dir, image_path)
                save_image(img, full_path, store_prompt, prompt)
                output_image_paths.append(image_path)

        return ImageGen(
            images=output_images,
            image_paths=output_image_paths,
            usages=[
                Usage(
                    prompt_tokens=usage_metadata.get("prompt_tokens", -1),
                    completion_tokens=usage_metadata.get("completion_tokens", -1),
                )
            ],
        )

    def _build_request(self, prompt, imgs, resize_inputs, temperature, aspect_ratio):
        """Build API request (OpenAI-compatible format)."""
        content = []

        # Add text prompt if provided
        if prompt:
            content.append({"type": "text", "text": prompt.strip()})

        # Add images if provided
        if imgs:
            # Ensure imgs is a list
            if isinstance(imgs, (str, Image.Image)):
                imgs = [imgs]

            for img in imgs:
                img_b64 = img_to_b64(img, resize_inputs)
                # OpenRouter expects data URL format for base64 images
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                })

        query_params = {
            "model": f"google/{self.model}",
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            # Note: OpenRouter may not support aspect_ratio the same way
            # Including it as metadata for now
            "provider": {
                "order": ["Google AI Studio"]
            }
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        api_url = "https://openrouter.ai/api/v1/chat/completions"

        return query_params, headers, api_url

    def _generate_multiple(self, n: int, **kwargs) -> "ImageGen":
        """Helper to generate multiple images by accumulating results."""
        n = kwargs.pop("n")
        result = None
        for _ in range(n):
            gen_result = self.generate(n=1, **kwargs)
            if result is None:
                result = gen_result
            else:
                result += gen_result
        return result


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ImageGen:
    images: List[Image.Image] = field(default_factory=list)
    image_paths: List[str] = field(default_factory=list)
    usages: List[Usage] = field(default_factory=list)

    @property
    def image(self) -> Optional[Image.Image]:
        return self.images[0] if self.images else None

    @property
    def image_path(self) -> Optional[str]:
        return self.image_paths[0] if self.image_paths else None

    @property
    def usage(self) -> Optional[Usage]:
        return self.usages[0] if self.usages else None

    def __add__(self, other: "ImageGen") -> "ImageGen":
        if isinstance(other, ImageGen):
            return ImageGen(
                images=self.images + other.images,
                image_paths=self.image_paths + other.image_paths,
                usages=self.usages + other.usages,
            )
        raise TypeError("Can only add ImageGen instances.")
