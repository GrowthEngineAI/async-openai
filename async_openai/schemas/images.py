from typing import Optional, Type, Any, Union, List, Dict
from lazyops.types import validator, lazyproperty, BaseModel

from async_openai.types.options import ImageSize, ImageFormat
from async_openai.types.resources import BaseResource
from async_openai.types.responses import BaseResponse
from async_openai.types.routes import BaseRoute

from fileio import File, FileType

__all__ = [
    'ImageData',
    'ImageObject',
    'ImageResponse',
    'ImageRoute',
]

class ImageData(BaseModel):
    url: Optional[str] = None
    data: Optional[bytes] = None

class ImageObject(BaseResource):
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
    
    def get_params(self, **kwargs) -> List:
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
    
    async def async_get_params(self, **kwargs) -> List:
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


class ImageResponse(BaseResponse):
    data: Optional[List[ImageData]]
    data_model: Optional[Type[ImageData]] = ImageData

    @lazyproperty
    def image_urls(self) -> List[str]:
        """
        Returns the list of image urls
        """
        if self.data:
            return [data.url for data in self.data] if self.data else []
        return None
    


class ImageRoute(BaseRoute):
    input_model: Optional[Type[BaseResource]] = ImageObject
    response_model: Optional[Type[BaseResource]] = ImageResponse

    @lazyproperty
    def api_resource(self):
        return 'images'

    def create(
        self, 
        input_object: Optional[Type[BaseResource]] = None,
        **kwargs
    ) -> ImageResponse:
        """
        
        """
        return super().create(input_object = input_object, **kwargs)
    
    async def async_create(
        self, 
        input_object: Optional[Type[BaseResource]] = None,
        **kwargs
    ) -> ImageResponse:
        """
        
        """
        return await super().async_create(input_object = input_object, **kwargs)

