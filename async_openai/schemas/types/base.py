
import json
import httpx
import datetime

from enum import Enum
from pydantic import validator
from pydantic.types import ByteSize
from async_openai.types import BaseModel, lazyproperty
from typing import List, Any, Optional, Type, Dict, Union, Iterator, cast, Callable
from async_openai.utils import logger
from async_openai.utils.helpers import is_coro_func

from async_openai.schemas.types.exceptions import error_handler

from fileio import File, FileType

__all__ = [
    'BaseEndpoint',
    'Permission',
    'Usage',
    'FilePurpose',
    'FileRequest',
    'FileData',
    'EventResult',
    'EngineResult',
    'BaseResult',
]

class Method(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"

class BaseEndpoint(BaseModel):
    method: Method
    url: str
    data: Optional[Union[Dict, Callable, Any]]
    params: Optional[Union[Dict, Callable, Any]]
    files: Optional[Union[Dict, Callable, Any]]
    sync_files: Optional[Union[Dict, Callable, Any]]

    async def async_get_params(
        self,
        **kwargs
    ):
        params = {
            "method": self.method.value,
            "url": self.url,
        }
        if self.data:
            if isinstance(self.data, dict):
                params["json"] = self.data
            elif isinstance(self.data, bytes):
                params["data"] = self.data
            elif is_coro_func(self.data):
                params["json"] = await self.data(**kwargs)
            elif callable(self.data):
                params["json"] = self.data(**kwargs)
            else:
                params["json"] = self.data
        if self.params:
            if isinstance(self.params, dict):
                params["params"] = self.params
            elif is_coro_func(self.params):
                params["params"] = await self.params(**kwargs)
            elif callable(self.params):
                params["params"] = self.params(**kwargs)
            else:
                params["params"] = self.params
        if self.files:
            if is_coro_func(self.files):
                params["files"] = await self.files(**kwargs)
            elif callable(self.files):
                params["files"] = self.files(**kwargs)
            else:
                params["files"] = self.files
        elif self.sync_files:
            if is_coro_func(self.sync_files):
                params["files"] = await self.sync_files(**kwargs)
            elif callable(self.sync_files):
                params["files"] = self.sync_files(**kwargs)
            else:
                params["files"] = self.sync_files
        return params

    def get_params(
        self,
        **kwargs
    ):
        params = {
            "method": self.method.value,
            "url": self.url,
        }
        if self.data:
            if isinstance(self.data, dict) or not isinstance(self.data, bytes) and not callable(self.data):
                params["json"] = self.data
            elif isinstance(self.data, bytes):
                params["data"] = self.data
            else:
                params["json"] = self.data(**kwargs)
        if self.params:
            if isinstance(self.params, dict) or not callable(self.params):
                params["params"] = self.params
            else:
                params["params"] = self.params(**kwargs)
        if self.sync_files:
            if callable(self.sync_files):
                params["files"] = self.sync_files(**kwargs)
            else:
                params["files"] = self.sync_files
        elif self.files:
            params["files"] = self.files(**kwargs) if callable(self.files) else self.files
        return params


class Permission(BaseModel):
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

class Usage(BaseModel):
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]


class FilePurpose(str, Enum):
    """
    File Purpose
    """

    finetune = "fine-tune"
    fine_tune = "fine-tune"
    train = "fine-tune-train"
    search = "search"

    @classmethod
    def parse_str(cls, value: str):
        if "train" in value:
            return cls.train
        elif "finetune" in value:
            return cls.finetune
        elif "fine-tune" in value:
            return cls.fine_tune
        elif "search" in value:
            return cls.search
        raise ValueError(f"Cannot convert {value} to FilePurpose")


class FileRequest(BaseModel):
    file_id: Optional[str]
    file: Optional[Union[str, FileType, Any]]
    filename: Optional[str] = None
    purpose: FilePurpose = FilePurpose.fine_tune
    model: Optional[str]

    @validator("purpose")
    def validate_purpose(cls, value):
        return FilePurpose.parse_str(value) if isinstance(value, str) else value
    
    def get_file_request_params(self, **kwargs) -> Dict:
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
    
    async def async_get_file_request_params(self, **kwargs) -> Dict:
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

    @lazyproperty
    def list_files_endpoint(self) -> BaseEndpoint:
        return BaseEndpoint(
            method = Method.GET,
            url = '/files',
        )
    
    @lazyproperty
    def upload_file_endpoint(self) -> BaseEndpoint:
        return BaseEndpoint(
            method = Method.POST,
            url = '/files',
            files = self.async_get_file_request_params,
            sync_files = self.get_file_request_params
        )

    @property
    def delete_file_endpoint(self) -> BaseEndpoint:
        return BaseEndpoint(
            method = Method.DELETE,
            url = f'/files/{self.file_id}'
        )

    @property
    def retrieve_file_endpoint(self) -> BaseEndpoint:
        return BaseEndpoint(
            method = Method.GET,
            url = f'/files/{self.file_id}'
        )

    @property
    def retrieve_file_content_endpoint(self) -> BaseEndpoint:
        return BaseEndpoint(
            method = Method.GET,
            url = f'/files/{self.file_id}/content'
        )


class FileData(BaseModel):
    id: str
    object: Optional[str] = 'file'
    bytes: Optional[ByteSize]
    created_at: Optional[datetime.datetime]
    filename: Optional[str]
    purpose: Optional[FilePurpose] = FilePurpose.fine_tune

    @validator("created_at")
    def validate_created_at(cls, value):
        return datetime.datetime.fromtimestamp(value, datetime.timezone.utc) if value else value
    

class EventResult(BaseModel):
    object: Optional[str]
    created_at: Optional[datetime.datetime]
    level: Optional[str]
    message: Optional[str]

    @property
    def since_seconds(self) -> int:
        if self.created_at is None: return -1
        return (datetime.datetime.now(datetime.timezone.utc) - self.created_at).total_seconds()


class EngineResult(BaseModel):
    id: Optional[str]
    object: Optional[str]
    owner: Optional[str]
    ready: Optional[bool]


class BaseResult(BaseModel):
    id: Optional[str]
    object: Optional[str]
    created: Optional[datetime.datetime]
    updated: Optional[datetime.datetime]
    choices: Optional[List[Any]]
    data: Optional[Union[Any, List[Any]]] # Used for single retrieval if not in list.
    events: Optional[List[Any]]

    usage: Optional[Usage] = None
    model: Optional[str] = None

    _request: Optional[Type[BaseModel]] = None
    _raw_request: Optional[httpx.Request] = None
    _raw_response: Optional[httpx.Response] = None

    _choice_model: Optional[Type[BaseModel]] = None
    _data_model: Optional[Type[BaseModel]] = None
    _event_model: Optional[Type[BaseModel]] = None

    @property
    def metadata_fields(self):
        # Default Metadata fields
        return [
            'id',
            'object',
            'created',
            # 'choices',
            'model',
            'usage',
        ]

    @property
    def since_seconds(self) -> int:
        return (datetime.datetime.now(datetime.timezone.utc) - self.created).total_seconds() \
            if self.created else -1

    @property
    def request_headers(self) -> Dict[str, str]:
        return self._raw_request.headers if self._raw_request else {}
    
    @property
    def response_headers(self) -> Dict[str, str]:
        return self._raw_response.headers if self._raw_response else {}

    @property
    def is_stream(self):
        return "text/event-stream" in self.response_headers.get("content-type", "")
    
    @property
    def request_id(self) -> Optional[str]:
        return self.response_headers.get("request-id", self.response_headers.get("x-request-id"))
    
    @property
    def organization(self) -> str:
        return self.response_headers.get("openai-organization")

    @property
    def response_ms(self) -> Optional[int]:
        h = self.response_headers.get("openai-processing-ms")
        return None if h is None else round(float(h))
    
    @property
    def response_timestamp(self) -> Optional[datetime.datetime]:
        dt = self.response_headers.get("date")
        return None if dt is None else datetime.datetime.strptime(dt, '%a, %d %b %Y %H:%M:%S GMT')

    @property
    def openai_version(self) -> Optional[str]:
        return self.response_headers.get("openai-version")
    
    @property
    def openai_model(self) -> Optional[str]:
        return self.response_headers.get("openai-model")
    
    def __len__(self):
        return len(self.choices) if self.choices else 0
    
    def __iter__(self):
        return iter(self.choices) if self.choices else iter([])
    
    def __getitem__(self, idx: int) -> Union[Type[BaseModel], Dict, Any]:
        return self.choices[idx] if self.choices else None

    def parse_stream(self, response: httpx.Response) -> Iterator[Dict]:
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
    
    def set_obj_meta(
        self,
        item: Dict
    ):
        """
        Sets the appropriate metadata fields
        """
        for key in self.metadata_fields:
            if key in item:
                if key == 'usage':
                    self.usage = Usage(**item['usage'])
                elif key in {'created', 'created_at'}:
                    self.created = datetime.datetime.fromtimestamp(item[key], datetime.timezone.utc)
                elif key in {'updated', 'updated_at'}:
                    self.updated = datetime.datetime.fromtimestamp(item[key], datetime.timezone.utc)
                elif key in {'result_files', 'training_files', 'validation_files'}:
                    vals = [
                        FileData.parse_obj(val) for val in item[key]
                    ]
                    setattr(self, key, vals)

                else: setattr(self, key, cast(type(getattr(self, key)), item[key]))
        
    def handle_data_item(
        self,
        item: Dict,
    ):
        """
        The handler function for each data item
        """
        if item.get('choices'):
            if self.choices is None: self.choices = []
            choices = item.get('choices', [])
            if self._choice_model:
                choices = [self._choice_model.parse_obj(choice) for choice in choices]
            self.choices.extend(choices)
        elif item.get('data'):
            if self.data is None: self.data = []
            data = item.get('data', [])
            if self._data_model:
                data = [self._data_model.parse_obj(d) for d in data]
            self.data.extend(data)
        elif item.get('events'):
            if self.events is None: self.events = []
            events = item.get('events', [])
            if self._event_model:
                events = [self._event_model.parse_obj(e) for e in events]
            self.events.extend(events)

        else:
            self.data = self._data_model.parse_obj(item) if self._data_model else item


    def handle_response(
        self,
    ):
        """
        The handler function for the response
        """
        if self.is_stream:
            has_set_meta = False
            for item in self.parse_stream(
                self._raw_response
            ):
            
                if "error" in item:
                    # handle error
                    raise error_handler(
                        response = self._raw_response,
                        stream = True,
                        data = item,
                    )
                if not has_set_meta:
                    # Set the initial params that are created from the first one.
                    self.set_obj_meta(item)
                    has_set_meta = True
                self.handle_data_item(item)
        else:
            item = self._raw_response.json()
            self.set_obj_meta(item)
            self.handle_data_item(item)

    def parse_result(
        self,
        raw_response: Optional[httpx.Response] = None,
        raw_request: Optional[httpx.Request] = None,
    ):
        if raw_response: self._raw_response = raw_response
        if raw_request: self._raw_request = raw_request
        self.handle_response()

