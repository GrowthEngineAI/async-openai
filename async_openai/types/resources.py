import json
import aiohttpx
import datetime
from pydantic.types import ByteSize
from lazyops.types import BaseModel, validator, lazyproperty
from lazyops.utils import ObjectDecoder
from async_openai.utils.logs import logger
from async_openai.utils.helpers import aparse_stream, parse_stream
from fileio import File, FileType

from async_openai.types.options import FilePurpose

from typing import Dict, Optional, Any, List, Type, Union, Tuple, Iterator, AsyncIterator, TYPE_CHECKING

__all__ = [
    'BaseResource',
    'Permission',
    'Usage',
    'FileObject',
    'EventObject',
    'FileResource',
]

VALID_SEND_KWARGS = [
    'method',
    'url',
    'content',
    'data',
    'files',
    'json',
    'params',
    'headers',
    'cookies',
    'auth',
    'follow_redirects',
    'timeout',
    'extensions',
]


class Usage(BaseModel):
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]

    @lazyproperty
    def consumption(self):
        return self.total_tokens
    
    def update(self, usage: Union['Usage', Dict[str, int]]):
        """
        Updates the consumption
        """
        for key in {
            'prompt_tokens',
            'completion_tokens',
            'total_tokens',
        }:
            if not hasattr(self, key):
                setattr(self, key, 0)
            val = usage.get(key, 0) if isinstance(usage, dict) else getattr(usage, key, 0)
            setattr(self, key, getattr(self, key) + val)




class BaseResource(BaseModel):

    """
    Base Object class for resources to
    inherit from
    """

    if TYPE_CHECKING:
        id: Optional[str]
        file_id: Optional[str]
        fine_tune_id: Optional[str]
        model_id: Optional[str]
        completion_id: Optional[str]
        openai_id: Optional[str]

    @lazyproperty
    def resource_id(self):
        """
        Returns the resource id
        """
        if hasattr(self, 'id'):
            return self.id
        if hasattr(self, 'file_id'):
            return self.file_id
        if hasattr(self, 'fine_tune_id'):
            return self.fine_tune_id
        if hasattr(self, 'model_id'):
            return self.model_id
        if hasattr(self, 'completion_id'):
            return self.completion_id
        return self.openai_id if hasattr(self, 'openai_id') else None
    
    @staticmethod
    def create_resource(
        resource: Type['BaseResource'],
        **kwargs
    ) -> Tuple[Type['BaseResource'], Dict]:
        """
        Extracts the resource from the kwargs and returns the resource 
        and the remaining kwargs
        """
        resource_fields = [field.name for field in resource.__fields__.values()]
        resource_kwargs = {k: v for k, v in kwargs.items() if k in resource_fields}
        return_kwargs = {k: v for k, v in kwargs.items() if k not in resource_fields}
        resource_obj = resource.parse_obj(resource_kwargs)
        return resource_obj, return_kwargs
    
    @classmethod
    def create_many(cls, data: List[Dict]) -> List['BaseResource']:
        """
        Creates many resources
        """
        # logger.info(f"Creating: {data}")
        return [cls.parse_obj(d) for d in data]
    
    @staticmethod
    def handle_json(
        content: Any,
        **kwargs
    ) -> Union[Dict, List]:
        """
        Handles the json response
        """
        return json.loads(content, cls = ObjectDecoder, **kwargs)


    @staticmethod
    def handle_stream(
        response: aiohttpx.Response
    ) -> Iterator[Dict]:
        """
        Handles the stream response
        """
        for line in parse_stream(response):
            if not line.strip(): continue
            try:
                yield json.loads(line)
            except Exception as e:
                logger.error(f'Error: {line}: {e}')
    
    @staticmethod
    async def ahandle_stream(
        response: aiohttpx.Response
    ) -> AsyncIterator[Dict]:
        """
        Handles the stream response
        """
        async for line in aparse_stream(response):
            if not line.strip(): continue
            try:
                yield json.loads(line)
            except Exception as e:
                logger.error(f'Error: {line}: {e}')


class Permission(BaseResource):
    id: str
    object: str
    created: datetime.datetime
    allow_create_engine: bool
    allow_sampling: bool
    allow_logprobs: bool
    allow_search_indices: bool
    allow_view: bool
    allow_fine_tuning: bool
    organization: str
    group: Optional[str]
    is_blocking: bool

    @property
    def since_seconds(self):
        return (datetime.datetime.now(datetime.timezone.utc) - self.created).total_seconds()


class FileObject(BaseResource):
    id: str
    object: Optional[str] = 'file'
    bytes: Optional[ByteSize]
    created_at: Optional[datetime.datetime]
    filename: Optional[str]
    purpose: Optional[FilePurpose] = FilePurpose.fine_tune

    @validator("created_at")
    def validate_created_at(cls, value):
        return datetime.datetime.fromtimestamp(value, datetime.timezone.utc) if value else value
    
    @classmethod
    def create_many(cls, data: List[Dict]) -> List['FileObject']:
        return [cls.parse_obj(d) for d in data]

class EventObject(BaseResource):
    object: Optional[str]
    created_at: Optional[datetime.datetime]
    level: Optional[str]
    message: Optional[str]

    @property
    def since_seconds(self) -> int:
        if self.created_at is None: return -1
        return (datetime.datetime.now(datetime.timezone.utc) - self.created_at).total_seconds()


class FileResource(BaseResource):
    file: Optional[Union[str, FileType, Any]]
    file_id: Optional[str]
    filename: Optional[str] = None
    purpose: FilePurpose = FilePurpose.fine_tune
    model: Optional[str]

    @validator("purpose")
    def validate_purpose(cls, value):
        return FilePurpose.parse_str(value) if isinstance(value, str) else value
    
    def get_params(self, **kwargs) -> List:
        """
        Transforms the data to the req params
        """
        files = [("purpose", (None, self.purpose.value))]
        if self.purpose == FilePurpose.search and self.model:
            files.append(("model", (None, self.model)))
        if self.file:
            file = File(self.file)
            files.append(
                ("file", (self.filename or file.name, file.read_bytes(), "application/octet-stream"))
            )
        return files
    
    async def async_get_params(self, **kwargs) -> List:
        """
        Transforms the data to the req params
        """
        files = [("purpose", (None, self.purpose.value))]
        if self.purpose == FilePurpose.search and self.model:
            files.append(("model", (None, self.model)))
        if self.file:
            file = File(self.file)
            files.append(
                ("file", (self.filename or file.name, await file.async_read_bytes(), "application/octet-stream"))
            )
        return files