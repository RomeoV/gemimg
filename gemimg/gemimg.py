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
    provider: str = "google"  # "google" or "openrouter"

    def __post_init__(self):
        # Determine API key based on provider
        if self.api_key is None:
            if self.provider == "openrouter":
                self.api_key = os.getenv("OPENROUTER_API_KEY")
                if not self.api_key:
                    raise ValueError(
                        "OPENROUTER_API_KEY environment variable is required when using provider='openrouter'. "
                        "Set it or pass api_key parameter."
                    )
            elif self.provider == "google":
                self.api_key = os.getenv("GEMINI_API_KEY")
                if not self.api_key:
                    raise ValueError(
                        "GEMINI_API_KEY environment variable is required when using provider='google'. "
                        "Set it or pass api_key parameter."
                    )
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

        if not self.api_key:
            raise ValueError(
                "API key is required. Set OPENROUTER_API_KEY or GEMINI_API_KEY environment variable, "
                "or pass it as `api_key` parameter."
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

        if self.provider == "openrouter":
            query_params, headers, api_url = self._build_openrouter_request(
                prompt, imgs, resize_inputs, temperature, aspect_ratio
            )
        else:  # google
            query_params, headers, api_url = self._build_google_request(
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

        if self.provider == "openrouter":
            return self._parse_openrouter_response(
                response_data, save, save_dir, webp, store_prompt, prompt
            )
        else:  # google
            return self._parse_google_response(
                response_data, save, save_dir, webp, store_prompt, prompt
            )

    def _parse_google_response(self, response_data, save, save_dir, webp, store_prompt, prompt):
        """Parse response from Google's Gemini API."""
        usage_metadata = response_data.get("usageMetadata", {})

        # Check for prohibited content
        candidates = response_data["candidates"][0]
        finish_reason = candidates.get("finishReason")
        if finish_reason in ["PROHIBITED_CONTENT", "NO_IMAGE"]:
            logger.error(f"Image was not generated due to {finish_reason}.")
            return None

        if "content" not in candidates:
            logger.error("No image is present in the response.")
            return None

        response_parts = candidates["content"]["parts"]
        output_images = []

        # Parse response parts for text and images
        for part in response_parts:
            if "inlineData" in part:
                output_images.append(b64_to_img(part["inlineData"]["data"]))

        output_image_paths = []
        if save:
            response_id = response_data.get("responseId", "output")
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
                    prompt_tokens=usage_metadata.get("promptTokenCount", -1),
                    completion_tokens=usage_metadata.get("candidatesTokenCount", -1),
                )
            ],
        )

    def _parse_openrouter_response(self, response_data, save, save_dir, webp, store_prompt, prompt):
        """Parse response from OpenRouter API."""
        # OpenRouter uses OpenAI-style response format
        usage_metadata = response_data.get("usage", {})

        choices = response_data.get("choices", [])
        if not choices:
            logger.error("No choices in OpenRouter response.")
            logger.error(f"Full response: {response_data}")
            return None

        message = choices[0].get("message", {})

        output_images = []

        # OpenRouter returns images in a separate 'images' field
        images_data = message.get("images", [])

        if not images_data:
            logger.error("No images field in OpenRouter response.")
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
            logger.error("No images found in OpenRouter response.")
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

    def _build_google_request(self, prompt, imgs, resize_inputs, temperature, aspect_ratio):
        """Build request for Google's Gemini API."""
        parts = []

        if imgs:
            # Ensure imgs is a list
            if isinstance(imgs, (str, Image.Image)):
                imgs = [imgs]

            img_b64_strings = [img_to_b64(img, resize_inputs) for img in imgs]
            parts.extend([img_b64_part(b64_str) for b64_str in img_b64_strings])

        if prompt:
            parts.append({"text": prompt.strip()})

        query_params = {
            "generationConfig": {
                "temperature": temperature,
                "imageConfig": {"aspectRatio": _validate_aspect(aspect_ratio)},
                "responseModalities": ["Image"],
            },
            "contents": [{"parts": parts}],
        }

        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

        return query_params, headers, api_url

    def _build_openrouter_request(self, prompt, imgs, resize_inputs, temperature, aspect_ratio):
        """Build request for OpenRouter API."""
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
