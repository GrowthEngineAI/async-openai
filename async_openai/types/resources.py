import json
import aiohttpx
import datetime
from pydantic.types import ByteSize
from lazyops.types import BaseModel, validator, lazyproperty
from lazyops.utils import ObjectDecoder
from async_openai.utils.logs import logger
from fileio import File, FileType

from async_openai.types.options import FilePurpose
from typing import Dict, Optional, Any, List, Type, Union, Tuple, Iterator

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

class BaseResource(BaseModel):

    """
    Base Object class for resources to
    inherit from
    """

    @lazyproperty
    def resource_id(self):
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
        return [cls.parse_obj(d) for d in data]
    
    @staticmethod
    def handle_json(
        content: Any,
        **kwargs
    ) -> Union[Dict, List]:
        return json.loads(content, cls = ObjectDecoder, **kwargs)


    @staticmethod
    def handle_stream(
        response: aiohttpx.Response
    ) -> Iterator[Dict]:
        for line in response.iter_lines():
            if not line: continue
            if "data: [DONE]" in line:
                # return here will cause GeneratorExit exception in urllib3
                # and it will close http connection with TCP Reset
                continue
            if line.startswith("data: "):
                line = line[len("data: ") :]
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