
import datetime
import aiohttpx
import contextlib
import functools
from lazyops.types import BaseModel, lazyproperty
from lazyops.types.models import get_pyd_field_names, Field, PYD_VERSION
from pydantic import RootModel
from typing import Dict, Optional, Any, List, Type, Union, Generator, AsyncGenerator, cast
from async_openai.types.errors import error_handler
from async_openai.types.resources import BaseResource, FileObject, Usage
from async_openai.utils import logger

__all__ = [
    'BaseResponse',
    'BaseBatchResponseItem',
    'BatchBaseResponse',
]


"""
Response Class
"""



if PYD_VERSION == 2:
    from pydantic import PrivateAttr, field_validator as _validator
    prevalidator = functools.partial(_validator, mode = 'before')
else:
    from pydantic import validator as _validator
    prevalidator = functools.partial(_validator, pre = True)


class BaseResponse(BaseResource):
    
    """
    Base Object class for responses to
    inherit from
    """

    id: Optional[str] = None
    object: Optional[str] = None
    created: Optional[datetime.datetime] = None
    updated: Optional[datetime.datetime] = None
    
    choices: Optional[List[Type[BaseResource]]] = None
    data: Optional[Union[Type[BaseResource], List[BaseResource]]] = None # Used for single retrieval if not in list.
    events: Optional[List[BaseResource]] = None

    data_model: Optional[Type[BaseResource]] = None
    choice_model: Optional[Type[BaseResource]] = None
    event_model: Optional[Type[BaseResource]] = None

    # Other Metadata
    usage: Optional[Usage] = None
    model: Optional[str] = None

    input_object: Optional[BaseResource] = Field(exclude = True)
    response: Optional[aiohttpx.Response] = Field(exclude = True)
    response_data: Optional[Union[Dict, List, Any]] = Field(default = None, exclude = True)

    if PYD_VERSION == 2:
        _has_metadata: Optional[bool] = PrivateAttr(default = False)
        _stream_consumed: Optional[bool] = PrivateAttr(default = False)
        _stream_chunks: Optional[List[Any]] = PrivateAttr(default = None)
    else:
        _has_metadata: Optional[bool] = False
        _stream_consumed: Optional[bool] = False
        _stream_chunks: Optional[List[Any]] = None


    @property
    def has_choices(self):
        """
        Returns whether the response has choices
        """
        return self.choice_model is not None
    
    @property
    def has_data(self):
        """
        Returns whether the response has data
        """
        return self.data_model is not None
    
    @property
    def has_events(self):
        """
        Returns whether the response has events
        """
        return self.event_model is not None
    
    @lazyproperty
    def resource_model(self) -> Type[BaseResource]:
        """
        Returns the appropriate resource model
        """
        if self.has_choices:
            return self.choice_model
        if self.has_data:
            return self.data_model
        return self.event_model if self.has_events else None

    @lazyproperty
    def excluded_params(self) -> List[str]:
        return [
            "data_model", "choice_model", "event_model", 
            "excluded_params", "resource_model", "input_model", "metadata_fields",
            "_stream_consumed", "_stream_chunks", "_has_metadata", 
            "response", "response_data", "input_object",
        ]
    
    def dict(self, *args, exclude: Any = None, **kwargs):
        """
        Returns the dict representation of the response
        """
        if exclude is None: exclude = set()
        exclude = set(exclude) | set(self.excluded_params)
        return super().dict(*args, exclude = exclude, **kwargs)


    """
    Metadata Properties
    """
    @lazyproperty
    def metadata_fields(self) -> List[str]:
        """
        Returns the metadata fields
        """
        return [
            name for name in get_pyd_field_names(self) if \
                name not in [
                    "response", "response_data", "input_object",
                    "data_model",  "choice_model", "event_model",
                    "data", "choices", "events"
                ]
        ]


    @property
    def since_seconds(self) -> int:
        """
        Returns the number of seconds since the response was created
        """
        return (datetime.datetime.now(datetime.timezone.utc) - self.created).total_seconds() \
            if self.created else -1
    
    @property
    def headers(self) -> Dict[str, str]:
        """
        Returns the response headers
        """
        return self.response.headers if self.response else {}
    
    @property
    def response_json(self) -> Dict:
        """
        Returns the response json
        """
        with contextlib.suppress(Exception):
            return self.response.json() if self.response else {}
        return BaseResource.handle_json(self.response.content) if self.response else {}

    @property
    def has_stream(self):
        """
        Returns whether the response is a stream
        """
        return "text/event-stream" in self.headers.get("content-type", "")
    
    @property
    def request_id(self) -> Optional[str]:
        """
        Returns the request id
        """
        return self.headers.get("request-id", self.headers.get("x-request-id"))
    
    @property
    def organization(self) -> str:
        """
        Returns the organization
        """
        return self.headers.get("openai-organization")
    
    @property
    def is_azure(self) -> bool:
        """
        Returns whether the response is from azure
        """
        return bool(self.headers.get("azureml-model-session", self.headers.get("azureml-model-group")))

    @property
    def response_ms(self) -> Optional[int]:
        """
        Returns the response time in ms
        """
        h = self.headers.get("openai-processing-ms")
        return None if h is None else round(float(h))
    
    @property
    def response_timestamp(self) -> Optional[datetime.datetime]:
        """
        Returns the response timestamp
        """
        dt = self.headers.get("date")
        return None if dt is None else datetime.datetime.strptime(dt, '%a, %d %b %Y %H:%M:%S GMT')

    @property
    def openai_version(self) -> Optional[str]:
        """
        Returns the openai version
        """
        return self.headers.get("openai-version")
    
    @property
    def openai_model(self) -> Optional[str]:
        """
        Returns the openai model
        """
        return self.headers.get("openai-model")
    
    @property
    def model_name(self) -> Optional[str]:
        """
        Returns the model name
        """
        return self.model.split('-')[1]
    
    """
    Object Data Properties
    """

    @property
    def resource_data(self) -> List[Union[Type[BaseResource], Dict, Any]]:
        """
        Returns the appropriate data object
        """
        if self.has_data: return self.data
        if self.has_choices: return self.choices
        return self.events if self.has_events else []
    

    def __len__(self):
        return len(self.resource_data)
    
    def __iter__(self):
        return iter(self.resource_data)
    
    def __getitem__(self, key: Union[str, int]) -> Union[Type[BaseModel], Dict, Any]:
        """
        Returns the appropriate data object from the response to support more
        native access to the response data
        """
        if isinstance(key, str):
            if not self.response_data:
                self.response_data = self.response.json()
            return self.response_data.get(key)
        return self.resource_data[key] if self.resource_data else None
    
    """
    Response Handling Methods
    """

    def handle_metadata(
        self,
        item: Dict,
        **kwargs
    ):
        """
        Sets the appropriate metadata fields
        for the response resource object
        """
        for field in self.metadata_fields:
            if field in item:
                if field == 'usage':
                    self.usage = Usage.parse_obj(item[field])
                    continue
                
                if 'created' in field or 'updated' in field:
                    # Handle created_at and updated_at fields as well
                    key = field.split('_', 1)[0]
                    setattr(self, key, datetime.datetime.fromtimestamp(item[field], datetime.timezone.utc))
                    continue

                if 'files' in field:
                    setattr(self, field, FileObject.create_many(item[field]))
                    continue

                setattr(self, field, cast(type(getattr(self, field)), item[field]))

        self._has_metadata = True

    def handle_data_item(
        self,
        item: Union[Dict, Any],
        **kwargs
    ):
        """
        Handle a single `data` item
        """
        if isinstance(item, dict) and item.get('data'):
            if self.data is None: self.data = []
            data = self.data_model.create_many(item['data'])
            self.data.extend(data)
        
        elif not self.data:
            self.data_model.parse_obj(item) if self.data_model else item

    
    def handle_choice_item(
        self,
        item: Union[Dict, Any],
        **kwargs
    ):
        """
        Handle a single `choice` item
        """
        if self.choices is None: self.choices = []
        
        if item.get('choices'):
            choices = self.choice_model.create_many(item['choices'])
            self.choices.extend(choices)
        elif self.has_stream:
            self.choices.append(self.choice_model.parse_obj(item))

        
    def handle_event_item(
        self,
        item: Union[Dict, Any],
        **kwargs
    ):
        """
        Handle a single `event` item
        """
        if item.get('events'):
            if self.events is None: self.events = []
            events = self.event_model.create_many(item['events'])
            self.events.extend(events)

    def handle_resource_item(
        self, 
        item: Union[Dict, Any],
        **kwargs
    ):
        """
        Handles a single resource item
        and routes it to the appropriate handler
        """
        if not self._has_metadata:
            self.handle_metadata(item, **kwargs)
        
        if self.has_choices and (item.get('choices') or self.has_stream):
            self.handle_choice_item(item, **kwargs)
        
        elif self.has_events and item.get('events'):
            self.handle_event_item(item, **kwargs)
        
        else:
            self.handle_data_item(item, **kwargs)
        
    def construct_resource(
        self, 
        parse_stream: Optional[bool] = True,
        **kwargs
    ):
        """
        Constructs the appropriate resource object
        from the response
        """
        if not self.has_stream: return self.handle_resource_item(item = self.response_json, **kwargs)
        if not parse_stream: return
        for item in self.handle_stream(
            response = self.response
        ):
            if "error" in item:
                raise error_handler(
                    response = self.response,
                    data = item,
                )
            
            self.handle_resource_item(item = item, **kwargs)
        self._stream_consumed = True


    async def aconstruct_resource(
        self, 
        parse_stream: Optional[bool] = True,
        **kwargs
    ):
        """
        Constructs the appropriate resource object
        from the response
        """
        if not self.has_stream: return self.handle_resource_item(item = self.response_json, **kwargs)
        if not parse_stream: return
        async for item in self.ahandle_stream(response=self.response):
            if "error" in item:
                raise error_handler(
                    response = self.response,
                    data = item,
                )
            
            self.handle_resource_item(item = item, **kwargs)
        self._stream_consumed = True
    
        
    @classmethod
    def prepare_response(
        cls, 
        response: aiohttpx.Response,
        input_object: 'BaseResource',
        parse_stream: Optional[bool] = True,
        **kwargs
    ) -> 'BaseResource':
        """
        Handles the response and returns the appropriate object
        """
        resource = cls(response = response, input_object = input_object)
        resource.construct_resource(parse_stream = parse_stream, **kwargs)
        return resource
    
    @classmethod
    async def aprepare_response(
        cls, 
        response: aiohttpx.Response,
        input_object: 'BaseResource',
        parse_stream: Optional[bool] = True,
        **kwargs
    ) -> 'BaseResource':
        """
        Handles the response and returns the appropriate object
        """
        resource = cls(response = response, input_object = input_object)
        await resource.aconstruct_resource(parse_stream = parse_stream, **kwargs)
        return resource


    def handle_stream_metadata(
        self,
        item: Dict[str, Any],
        **kwargs
    ):
        """
        Handles the stream metadata
        """
        if not self.id and item.get('id'):
            self.id = item['id']
        if not self.created and item.get('created'):
            self.created = datetime.datetime.fromtimestamp(item['created'], datetime.timezone.utc)
        if not self.usage:
            self.usage = Usage(completion_tokens=0)
    
    @lazyproperty
    def choices_results(self) -> List[Type[BaseResource]]:
        """
        Parses the choices first
        """
        if self.choices is None:
            self.construct_resource()
        return self.choices

    """
    New Handling of Stream Specific Responses
    """

    def parse_stream_item(self, item: Union[Dict, Any], **kwargs) -> Any:
        """
        Parses a single stream item
        """
        return item

    def stream(self, **kwargs) -> Generator[Dict, None, None]:
        """
        Handles the stream
        """
        if self._stream_consumed:
            yield from self._stream_chunks
            return
        self._stream_chunks = []
        for item in self.handle_stream(response = self.response, streaming = True):
            if "error" in item:
                raise error_handler(response = self.response, data = item)
            # self.handle_resource_item(item = item, **kwargs)
            stream_item = self.parse_stream_item(item = item, **kwargs)
            if stream_item is not None:
                self._stream_chunks.append(stream_item)
                yield stream_item
        self._stream_consumed = True
    
    async def astream(self, **kwargs) -> AsyncGenerator[Dict, None]:
        """
        Handles the stream
        """
        if self._stream_consumed:
            for stream_item in self._stream_chunks:
                yield stream_item
            return
        self._stream_chunks = []
        async for item in self.ahandle_stream(response = self.response, streaming = True):
            if "error" in item:
                raise error_handler(response = self.response, data = item)
            # self.handle_resource_item(item = item, **kwargs)
            stream_item = self.parse_stream_item(item = item, **kwargs)
            if stream_item is not None:
                self._stream_chunks.append(stream_item)
                yield stream_item
        self._stream_consumed = True
        

class BatchResponseData(BaseModel):
    """
    The response data for a batch request
    """
    status_code: Optional[int] = None
    request_id: Optional[str] = None
    body: Optional[Dict[str, Any]] = Field(default_factory = dict, description = "The body of the response")
    usage: Optional[Usage] = Field(default_factory = Usage, description = "The usage of the response")
    system_fingerprint: Optional[str] = None
    

class BaseBatchResponseItem(BaseResponse):
    """
    The response for a single item in a batch request
    """
    custom_id: Optional[str] = None
    input_object: Optional[BaseResource] = Field(None, exclude = True)
    response: Optional[BatchResponseData] = Field(default_factory = BatchResponseData, description = "The response for the item")
    error: Optional[Any] = Field(default = None, description = "The error for the item")

    if PYD_VERSION == 2:
        _extra: Optional[Dict[str, Any]] = PrivateAttr(default_factory = dict)
    else:
        _extra: Optional[Dict[str, Any]] = Field(default_factory = dict, description = "Extra data for the item", exclude = True)


    """
    Modified Properties to support batch requests
    """
        
    @property
    def has_stream(self):
        """
        Returns whether the response is a stream
        """
        return False
    
    @property
    def headers(self) -> Optional[Dict[str, str]]:
        """
        Returns the response headers
        """
        return self._extra.get('headers', {})

    @headers.setter
    def headers(self, value: Dict[str, str]):
        """
        Sets the response headers
        """
        self._extra['headers'] = value
    
    @property
    def response_json(self) -> Dict:
        """
        Returns the response json
        """
        return self.response.body

    @property
    def request_id(self) -> Optional[str]:
        """
        Returns the request id
        """
        return self.response.request_id
    
    @property
    def organization(self) -> str:
        """
        Returns the organization
        """
        return self.headers.get("openai-organization")
    
    @property
    def is_azure(self) -> bool:
        """
        Returns whether the response is from azure
        """
        return bool(self.headers.get("azureml-model-session", self.headers.get("azureml-model-group")))

    @property
    def response_ms(self) -> Optional[int]:
        """
        Returns the response time in ms
        """
        h = self.headers.get("openai-processing-ms")
        return None if h is None else round(float(h))
    
    @property
    def response_timestamp(self) -> Optional[datetime.datetime]:
        """
        Returns the response timestamp
        """
        dt = self.headers.get("date")
        return None if dt is None else datetime.datetime.strptime(dt, '%a, %d %b %Y %H:%M:%S GMT')

    @property
    def openai_version(self) -> Optional[str]:
        """
        Returns the openai version
        """
        return self.headers.get("openai-version")
    
    @property
    def openai_model(self) -> Optional[str]:
        """
        Returns the openai model
        """
        return self.headers.get("openai-model", self.response.body.get('model', None))
    
    @property
    def model_name(self) -> Optional[str]:
        """
        Returns the model name
        """
        return self.model.split('-')[1]


    """
    Object Data Properties
    """

    @property
    def resource_data(self) -> List[Union[Type[BaseResource], Dict, Any]]:
        """
        Returns the appropriate data object
        """
        if self.has_data: return self.data
        if self.has_choices: return self.choices
        return self.events if self.has_events else []
    
    
    def __getitem__(self, key: Union[str, int]) -> Union[Type[BaseModel], Dict, Any]:
        """
        Returns the appropriate data object from the response to support more
        native access to the response data
        """
        if isinstance(key, str):
            if not self.response_data:
                self.response_data = self.response.body
            return self.response_data.get(key)
        return self.resource_data[key] if self.resource_data else None
    
    """
    Response Handling Methods
    """

    def construct_resource(
        self, 
        parse_stream: Optional[bool] = False,
        **kwargs
    ):
        """
        Constructs the appropriate resource object
        from the response
        """
        return self.handle_resource_item(item = self.response_json, **kwargs)


    async def aconstruct_resource(
        self, 
        parse_stream: Optional[bool] = False,
        **kwargs
    ):
        """
        Constructs the appropriate resource object
        from the response
        """
        return self.handle_resource_item(item = self.response_json, **kwargs)
    
        
    @classmethod
    def prepare_response(
        cls, 
        response_item: Dict[str, Any],
        input_object: 'BaseResource',
        parse_stream: Optional[bool] = False,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> 'BaseBatchResponseItem':
        """
        Handles the response and returns the appropriate object
        """
        resource = cls.model_validate(response_item)
        resource.input_object = input_object
        if headers: resource.headers = headers
        resource.construct_resource(parse_stream = parse_stream, **kwargs)
        return resource
    
    @classmethod
    async def aprepare_response(
        cls, 
        response_item: Dict[str, Any],
        input_object: 'BaseResource',
        parse_stream: Optional[bool] = False,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> 'BaseBatchResponseItem':
        """
        Handles the response and returns the appropriate object
        """
        resource = cls.model_validate(response_item)
        resource.input_object = input_object
        if headers: resource.headers = headers
        await resource.aconstruct_resource(parse_stream = parse_stream, **kwargs)
        return resource



class BatchBaseResponse(RootModel):
    """
    The base response for batch requests
    """
    root: List[BaseBatchResponseItem]

    def __len__(self):
        """
        Returns the length of the root
        """
        return len(self.root)


SUCCESSFUL_STATUSES = [
    'completed',
]
UNSUCESSFUL_STATUSES = [
    'failed',
    'expired',
    'cancelled',
]
IN_PROGRESS_STATUSES = [
    'validating',
    'in_progress',
    'finalizing',
    'cancelling',
]

TERMINAL_STATUSES = SUCCESSFUL_STATUSES + UNSUCESSFUL_STATUSES


class BatchStatusResponse(BaseModel):
    """
    The response for a single item in a batch request
    """
    id: str = Field(..., description = "The ID of the batch request")
    object: str = Field(..., description = "The type of the batch request")
    endpoint: str = Field(..., description = "The endpoint of the batch request")
    errors: Optional[Any] = Field(None, description = "The errors of the batch request")
    input_file_id: Optional[str] = Field(None, description = "The input file ID of the batch request")
    completion_window: Optional[str] = Field(None, description = "The completion window of the batch request")
    status: Optional[str] = Field(None, description = "The status of the batch request")
    output_file_id: Optional[str] = Field(None, description = "The output file ID of the batch request")
    error_file_id: Optional[str] = Field(None, description = "The error file ID of the batch request")
    created_at: Optional[datetime.datetime] = Field(None, description = "The created at of the batch request")
    in_progress_at: Optional[datetime.datetime] = Field(None, description = "The in progress at of the batch request")
    expires_at: Optional[datetime.datetime] = Field(None, description = "The expires at of the batch request")
    completed_at: Optional[datetime.datetime] = Field(None, description = "The completed at of the batch request")
    failed_at: Optional[datetime.datetime] = Field(None, description = "The failed at of the batch request")
    expired_at: Optional[datetime.datetime] = Field(None, description = "The expired at of the batch request")
    request_counts: Optional[Dict[str, int]] = Field(None, description = "The request counts of the batch request")
    metadata: Optional[Dict[str, Any]] = Field(None, description = "The metadata of the batch request")

    request_idx_mapping: Dict[str, int] = Field(default_factory = dict, description = "The request index mapping of the batch request")

    @prevalidator('created_at', 'in_progress_at', 'expires_at', 'completed_at', 'failed_at', 'expired_at')
    def validate_datetime(cls, v: Optional[Union[int, datetime.datetime]]) -> Optional[datetime.datetime]:
        """
        Validates the datetime
        """
        if not v: return None
        if not isinstance(v, datetime.datetime): v = datetime.datetime.fromtimestamp(v, datetime.timezone.utc)
        return v

    @property
    def is_completed(self) -> bool:
        """
        Returns whether the batch request is completed
        """
        return self.status in TERMINAL_STATUSES

    @property
    def is_in_progress(self) -> bool:
        """
        Returns whether the batch request is in progress
        """
        return self.status in IN_PROGRESS_STATUSES

    @property
    def is_successful(self) -> bool:
        """
        Returns whether the batch request is successful
        """
        return self.status in SUCCESSFUL_STATUSES

    @property
    def is_unsuccessful(self) -> bool:
        """
        Returns whether the batch request is unsuccessful
        """
        return self.status in UNSUCESSFUL_STATUSES