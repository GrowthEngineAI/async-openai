
import datetime
import aiohttpx
import contextlib

from lazyops.types import BaseModel, lazyproperty
from typing import Dict, Optional, Any, List, Type, Union, cast
from async_openai.types.errors import error_handler
from async_openai.types.resources import BaseResource, FileObject, Usage

__all__ = [
    'BaseResponse'
]


"""
Response Class
"""

class BaseResponse(BaseResource):
    
    """
    Base Object class for responses to
    inherit from
    """

    id: Optional[str]
    object: Optional[str]
    created: Optional[datetime.datetime]
    updated: Optional[datetime.datetime]
    
    choices: Optional[List[Type[BaseResource]]]
    data: Optional[Union[Type[BaseResource], List[BaseResource]]] # Used for single retrieval if not in list.
    events: Optional[List[BaseResource]]

    data_model: Optional[Type[BaseResource]] = None
    choice_model: Optional[Type[BaseResource]] = None
    event_model: Optional[Type[BaseResource]] = None

    # Other Metadata
    usage: Optional[Usage] = None
    model: Optional[str] = None

    _input_object: Optional[Type[BaseResource]] = None
    _response: Optional[aiohttpx.Response] = None

    _has_metadata: Optional[bool] = False


    @property
    def has_choices(self):
        return self.choice_model is not None
    
    @property
    def has_data(self):
        return self.data_model is not None
    
    @property
    def has_events(self):
        return self.event_model is not None
    
    @lazyproperty
    def resource_model(self) -> Type[BaseResource]:
        if self.has_choices:
            return self.choice_model
        if self.has_data:
            return self.data_model
        return self.event_model if self.has_events else None

    """
    Metadata Properties
    """
    @lazyproperty
    def metadata_fields(self) -> List[str]:
        return [
            field.name for field in self.__fields__.values() if \
                field.name not in [
                    "_input_object", "_response", 
                    "data_model",  "choice_model", "event_model",
                    "data", "choices", "events"
                ]
        ]


    @property
    def since_seconds(self) -> int:
        return (datetime.datetime.now(datetime.timezone.utc) - self.created).total_seconds() \
            if self.created else -1
    
    @property
    def headers(self) -> Dict[str, str]:
        return self._response.headers if self._response else {}
    
    @property
    def response_json(self) -> Dict:
        with contextlib.suppress(Exception):
            return self._response.json() if self._response else {}
        return BaseResource.handle_json(self._response.content) if self._response else {}

    @property
    def has_stream(self):
        return "text/event-stream" in self.headers.get("content-type", "")
    
    @property
    def request_id(self) -> Optional[str]:
        return self.headers.get("request-id", self.headers.get("x-request-id"))
    
    @property
    def organization(self) -> str:
        return self.headers.get("openai-organization")

    @property
    def response_ms(self) -> Optional[int]:
        h = self.headers.get("openai-processing-ms")
        return None if h is None else round(float(h))
    
    @property
    def response_timestamp(self) -> Optional[datetime.datetime]:
        dt = self.headers.get("date")
        return None if dt is None else datetime.datetime.strptime(dt, '%a, %d %b %Y %H:%M:%S GMT')

    @property
    def openai_version(self) -> Optional[str]:
        return self.headers.get("openai-version")
    
    @property
    def openai_model(self) -> Optional[str]:
        return self.headers.get("openai-model")
    
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
    
    def __getitem__(self, idx: int) -> Union[Type[BaseModel], Dict, Any]:
        return self.resource_data[idx] if self.resource_data else None
    
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
            self.data = self.data_model.parse_obj(item) if self.data_model else item

    
    def handle_choice_item(
        self,
        item: Union[Dict, Any],
        **kwargs
    ):
        """
        Handle a single `choice` item
        """
        if item.get('choices'):
            if self.choices is None: self.choices = []
            choices = self.choice_model.create_many(item['choices'])
            self.choices.extend(choices)
        
    
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
        
        if self.has_choices and item.get('choices'):
            self.handle_choice_item(item, **kwargs)
        
        elif self.has_events and item.get('events'):
            self.handle_event_item(item, **kwargs)
        
        else:
            self.handle_data_item(item, **kwargs)
        
    
    def construct_resource(
        self, 
        **kwargs
    ):
        """
        Constructs the appropriate resource object
        from the response
        """
        if not self.has_stream:
            return self.handle_resource_item(item = self.response_json, **kwargs)
        
        for item in self.handle_stream(
            response = self._response
        ):
            if "error" in item:
                raise error_handler(
                    response = self._response,
                    data = item,
                )
            
            self.handle_resource_item(item = item, **kwargs)


    @classmethod
    def prepare_response(
        cls, 
        response: aiohttpx.Response,
        input_object: Type['BaseResource'],
        **kwargs
    ) -> Type['BaseResponse']:
        """
        Handles the response and returns the appropriate object
        """
        # logger.info(f"Preparing Response for {cls.__name__}")
        resource = cls(
            _response = response,
            _input_object = input_object,
        )
        resource.construct_resource(**kwargs)
        # logger.info(f"Response: {resource}")
        return resource
