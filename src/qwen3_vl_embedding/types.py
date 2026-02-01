from typing import Literal, Required, TypedDict

# ! Reference types from OpenAI SDK
# from openai.types.chat import (
#     ChatCompletionContentPartImageParam,
#     ChatCompletionContentPartTextParam,
# )


class ImageURL(TypedDict, total=False):
    url: Required[str]
    """Either a URL of the image or the base64 encoded image data."""

    detail: Literal["auto", "low", "high"]
    """Specifies the detail level of the image.

    Learn more in the
    [Vision guide](https://platform.openai.com/docs/guides/vision#low-or-high-fidelity-image-understanding).
    """


class ChatCompletionContentPartImageParam(TypedDict, total=False):
    """Learn about [image inputs](https://platform.openai.com/docs/guides/vision)."""

    image_url: Required[ImageURL]

    type: Required[Literal["image_url"]]
    """The type of the content part."""


class ChatCompletionContentPartTextParam(TypedDict, total=False):
    """
    Learn about [text inputs](https://platform.openai.com/docs/guides/text-generation).
    """

    text: Required[str]
    """The text content."""

    type: Required[Literal["text"]]
    """The type of the content part."""


class VideoURL(TypedDict):
    """A URL pointing to a video resource."""

    url: str
    """The URL of the video."""


class ChatCompletionContentPartVideoParam(TypedDict, total=False):
    """Addon content part representing a video URL."""

    video_url: Required[VideoURL]

    type: Required[Literal["video_url"]]
    """The type of the content part."""


EmbeddingContentPart = (
    ChatCompletionContentPartVideoParam
    | ChatCompletionContentPartTextParam
    | ChatCompletionContentPartImageParam
)
