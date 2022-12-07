from enum import Enum
from pydantic import validator
from typing import List, Any, Optional, Union, Dict, Type
from async_openai.types import BaseModel, lazyproperty
from async_openai.schemas.types.base import BaseResult, Method, BaseEndpoint

from fileio import File, FileType

__all__ = [
    'ImageSize',
    'ImageFormat',
    'ImageRequest',
    'ImageData',
    'ImageResult',
]

class ImageSize(str, Enum):
    """
    Size of the image
    """

    small = "256x256"
    medium = "512x512"
    large = "1024x1024"

    @classmethod
    def from_str(cls, value: str) -> "ImageSize":
        """
        :param value: Size of the image
        :type value: str
        :return: ImageSize
        :rtype: ImageSize
        """
        if value == "256x256":
            return cls.small
        if value == "512x512":
            return cls.medium
        if value == "1024x1024":
            return cls.large
        raise ValueError(f"Cannot convert {value} to ImageSize")

class ImageFormat(str, Enum):
    """
    Format of the image
    """

    url = "url"
    b64 = "b64_json"
    b64_json = "b64_json"


class ImageRequest(BaseModel):
    prompt: Optional[str]
    mask: Optional[Union[str, FileType, Any]]
    image: Optional[Union[str, FileType, Any]]
    n: Optional[int] = 1
    size: Optional[Union[str, ImageSize]] = ImageSize.large
    response_format: Optional[Union[str, ImageFormat]] = ImageFormat.url
    user: Optional[str] = None

    @validator("size")
    def validate_size(cls, value):
        return ImageSize.from_str(value) if isinstance(value, str) else value
    
    @validator("response_format")
    def validate_response_format(cls, value):
        if isinstance(value, str):
            value = ImageFormat(value)
        return value
    

    def get_file_request_params(self, **kwargs) -> Dict:
        """
        Transforms the data to the req params
        """
        files = [(k, (None, v)) for k, v in self.dict(exclude_none=True, exclude={'mask', 'image'}).items()]
        if self.mask:
            mask = File(self.mask)
            files.append(("mask", ("mask", mask.read_bytes(), "application/octet-stream")))
        if self.image:
            image = File(self.image)
            files.append(("image", ("image", image.read_bytes(), "application/octet-stream")))
        
        return files

    
    async def async_get_file_request_params(self, **kwargs) -> Dict:
        """
        Transforms the data to the req params
        """
        files = [(k, (None, v)) for k, v in self.dict(exclude_none=True, exclude={'mask', 'image'}).items()]
        if self.mask:
            mask = File(self.mask)
            files.append(("mask", ("mask", await mask.async_read_bytes(), "application/octet-stream")))
        if self.image:
            image = File(self.image)
            files.append(("image", ("image", await image.async_read_bytes(), "application/octet-stream")))
        
        return files

    
    @lazyproperty
    def create_image_endpoint(self) -> BaseEndpoint:
        return BaseEndpoint(
            method = Method.POST,
            url = '/images/generations',
            files = self.async_get_file_request_params,
            sync_files = self.get_file_request_params
        )
    
    @lazyproperty
    def create_image_edit_endpoint(self) -> BaseEndpoint:
        return BaseEndpoint(
            method = Method.POST,
            url = '/images/edits',
            files = self.async_get_file_request_params,
            sync_files = self.get_file_request_params
        )
    
    @lazyproperty
    def create_image_variation_endpoint(self) -> BaseEndpoint:
        return BaseEndpoint(
            method = Method.POST,
            url = '/images/variations',
            files = self.async_get_file_request_params,
            sync_files = self.get_file_request_params
        )
    

class ImageData(BaseModel):
    url: Optional[str] = None
    data: Optional[bytes] = None


class ImageResult(BaseResult):
    data: Optional[Union[ImageData, List[ImageData]]]
    _data_model: Optional[Type[ImageData]] = ImageData
    _request: Optional[ImageRequest] = None

    @property
    def metadata_fields(self):
        return [
            'created',
            'object',
            # 'data',
        ]

