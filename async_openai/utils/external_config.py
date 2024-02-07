import json
import logging
import pathlib
import aiohttpx
import contextlib
from typing import Optional, Dict, Union, Any, List, Type, TYPE_CHECKING
from lazyops.types import BaseSettings, validator, BaseModel, lazyproperty, Field, PYD_VERSION
from lazyops.libs.proxyobj import ProxyObject
from lazyops.libs.abcs.configs.types import AppEnv
from async_openai.version import VERSION
from async_openai.types.options import ApiType
from async_openai.types.context import ModelCostItem, ModelContextHandler

if PYD_VERSION == 2:
    from pydantic import model_validator
else:
    from lazyops.types.models import root_validator

if TYPE_CHECKING:
    from async_openai.types.routes import BaseRoute, BaseResource


preset_path = pathlib.Path(__file__).parent.joinpath('presets')


class ExternalProviderConfig(BaseModel):
    """
    External Provider - Configuration
    """
    api_base: str = Field(..., description="The base URL for the API")
    api_path: Optional[str] = Field(None, description="The path for the API")
    api_key_header: Optional[str] = Field(None, description="The header for the API Key")
    api_key_scheme: Optional[str] = Field(None, description="The scheme for the API Key")
    api_key: Optional[str] = Field(None, description="The API Key")

    custom_headers: Optional[Dict[str, str]] = Field(None, description="Custom Headers")
    proxy_url: Optional[str] = Field(None, description="The Proxy URL")
    proxy_headers: Optional[Dict[str, str]] = Field(None, description="Proxy Headers")
    hf_compatible: Optional[bool] = Field(None, description="Whether the provider is HuggingFace Compatible for Tokenization")

    @validator("custom_headers", "proxy_headers", pre=True)
    def validate_headers(cls, value: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
        """
        Validates the Headers
        """
        if value is None: return {}
        return {str(k).strip(): str(v).strip() for k,v in value.items()}

    @property
    def has_api_key(self) -> bool:
        """
        Returns whether the API Key is present
        """
        return self.api_key is not None
    
    @property
    def api_key_value(self) -> Optional[str]:
        """
        Returns the API Key Value
        """
        if not self.has_api_key: return None
        return f"{self.api_key_scheme} {self.api_key}" if self.api_key_scheme else self.api_key
    
    @property
    def has_proxy(self) -> bool:
        """
        Returns whether the Proxy is present
        """
        return self.proxy_url is not None
    
    @property
    def api_url(self) -> str:
        """
        Returns the API URL
        """
        if not self.api_path: return self.api_base
        return aiohttpx.urljoin(self.api_base, self.api_path)


class ProviderModel(ModelCostItem):
    """
    A Provider Model
    """
    name: str = Field(..., description="The Model Name")


class ProviderRoute(BaseModel):
    """
    A Provider Route
    """
    name: Optional[str] = Field(None, description="The Route Name")
    api_resource: Optional[str] = Field(None, description="The Route Path")
    root_name: Optional[str] = Field(None, description="The Root Name")
    # params: Optional[Dict[str, Type]] = Field(None, description="The Route Parameters and expected Types")

    if TYPE_CHECKING:
        route_class: Optional[Type['BaseRoute']] = Field(None, description="The Route Class")
        object_class: Optional[Type['BaseResource']] = Field(None, description="The Resource Class")
        response_class: Optional[Type['BaseResource']] = Field(None, description="The Response Class")
    else:
        route_class: Optional[Any] = Field(None, description="The Route Class")
        object_class: Optional[Any] = Field(None, description="The Resource Class")
        response_class: Optional[Any] = Field(None, description="The Response Class")
    

    @validator("route_class", "object_class", "response_class", pre=True)
    def validate_route_classes(cls, value: Union[str, Type]) -> Type:
        """
        Validates the Route Classes
        """
        if value is None: return None
        if isinstance(value, str):
            from lazyops.utils.lazy import lazy_import
            return lazy_import(value)
        return value


class ExternalProviderRoutes(BaseModel):
    """
    External Provider - Routes
    """
    completion: Optional[ProviderRoute] = Field(None, description="The Completion Route")
    chat: Optional[ProviderRoute] = Field(None, description="The Chat Route")
    embedding: Optional[ProviderRoute] = Field(None, description="The Embedding Route")
    # if TYPE_CHECKING:

    if PYD_VERSION == 2:
        @model_validator(mode = 'after')
        def validate_routes(self):
            """
            Validates the Routes
            """
            from async_openai.schemas import (
                ChatRoute, ChatResponse, ChatObject,
                CompletionRoute, CompletionResponse, CompletionObject,
                EmbeddingRoute, EmbeddingResponse, EmbeddingObject,
                # TODO - Add the rest of the routes
            )
            if self.chat is None:
                self.chat = ProviderRoute(
                    name = "chat",
                    route_class = ChatRoute,
                    object_class = ChatObject,
                    response_class = ChatResponse,
                )
            if not self.chat.name:
                self.chat.name = "chat"
            if self.chat.route_class is None:
                self.chat.route_class = ChatRoute
            if self.chat.object_class is None:
                self.chat.object_class = ChatObject
            if self.chat.response_class is None:
                self.chat.response_class = ChatResponse

            if self.completion is None:
                self.completion = ProviderRoute(
                    name = "completion",
                    route_class = CompletionRoute,
                    object_class = CompletionObject,
                    response_class = CompletionResponse,
                )
            if not self.completion.name:
                self.completion.name = "completion"
            if self.completion.route_class is None:
                self.completion.route_class = CompletionRoute
            if self.completion.object_class is None:
                self.completion.object_class = CompletionObject
            if self.completion.response_class is None:
                self.completion.response_class = CompletionResponse

            if self.embedding is None:
                self.embedding = ProviderRoute(
                    name = "embedding",
                    route_class = EmbeddingRoute,
                    object_class = EmbeddingObject,
                    response_class = EmbeddingResponse,
                )
            
            if not self.embedding.name:
                self.embedding.name = "embedding"
            if self.embedding.route_class is None:
                self.embedding.route_class = EmbeddingRoute
            if self.embedding.object_class is None:
                self.embedding.object_class = EmbeddingObject
            if self.embedding.response_class is None:
                self.embedding.response_class = EmbeddingResponse
            return self

    else:
        @root_validator()
        def validate_routes(cls, values: Dict[str, Any]) -> Dict[str, Any]:
            """
            Validates the Routes
            """
            from async_openai.schemas import (
                ChatRoute, ChatResponse, ChatObject,
                CompletionRoute, CompletionResponse, CompletionObject,
                EmbeddingRoute, EmbeddingResponse, EmbeddingObject,
                # TODO - Add the rest of the routes
            )
            if values.get('chat') is None:
                values['chat'] = ProviderRoute(
                    name = "chat",
                    route_class = ChatRoute,
                    object_class = ChatObject,
                    response_class = ChatResponse,
                )
            else:
                values['chat'] = ProviderRoute.parse_obj(values['chat'])
            if not values['chat'].name:
                values['chat'].name = "chat"
            if values['chat'].route_class is None:
                values['chat'].route_class = ChatRoute
            if values['chat'].object_class is None:
                values['chat'].object_class = ChatObject
            if values['chat'].response_class is None:
                values['chat'].response_class = ChatResponse
            
            if values.get('completion') is None:
                values['completion'] = ProviderRoute(
                    name = "completion",
                    route_class = CompletionRoute,
                    object_class = CompletionObject,
                    response_class = CompletionResponse,
                )
            else:
                values['completion'] = ProviderRoute.parse_obj(values['completion'])
            if not values['completion'].name:
                values['completion'].name = "completion"
            if values['completion'].route_class is None:
                values['completion'].route_class = CompletionRoute
            if values['completion'].object_class is None:
                values['completion'].object_class = CompletionObject
            if values['completion'].response_class is None:
                values['completion'].response_class = CompletionResponse
            
            if values.get('embedding') is None:
                values['embedding'] = ProviderRoute(
                    name = "embedding",
                    route_class = EmbeddingRoute,
                    object_class = EmbeddingObject,
                    response_class = EmbeddingResponse,
                )
            else:
                values['embedding'] = ProviderRoute.parse_obj(values['embedding'])
            if not values['embedding'].name:
                values['embedding'].name = "embedding"
            if values['embedding'].route_class is None:
                values['embedding'].route_class = EmbeddingRoute
            if values['embedding'].object_class is None:
                values['embedding'].object_class = EmbeddingObject
            if values['embedding'].response_class is None:
                values['embedding'].response_class = EmbeddingResponse
            return values
        
    @property
    def api_route_classes(self) -> Dict[str, Type['BaseRoute']]:
        """
        Returns the Route Classes
        """
        return {
            "chat": self.chat.route_class,
            "completion": self.completion.route_class,
            "embedding": self.embedding.route_class,
        }
            

class ExternalProviderSettings(BaseModel):
    """
    External Provider Configuration
    """
    name: str = Field(..., description="The Provider Name")
    config: ExternalProviderConfig = Field(default_factory = ExternalProviderConfig, description="The Provider Configuration")
    models: List[ProviderModel] = Field(default_factory=list, description="The Provider Models")
    routes: Optional[ExternalProviderRoutes] = Field(default_factory=ExternalProviderRoutes, description="The Provider Routes")

    @classmethod
    def from_preset(
        cls, 
        name: Optional[str] = None,
        path: Optional[Union[str, pathlib.Path]] = None,
        **overrides
    ) -> 'ExternalProviderSettings':
        """
        Loads the Provider Settings from a Preset
        """
        assert name or path, "You must provide either a name or a path to the preset"
        if path:
            preset_file = pathlib.Path(path)
            assert preset_file.exists(), f"Could not find the preset path: {preset_file}"
        else:
            preset_file = preset_path.joinpath(f"{name}.yaml")
            if not preset_file.exists():
                raise FileNotFoundError(f"Could not find the preset file: {preset_file} for {name}")
        
        assert preset_file.suffix in {
            ".yaml", ".yml", ".json"
        }, f"The preset file must be a YAML or JSON file: {preset_file}"
        
        from lazyops.libs.abcs.utils.envvars import parse_envvars_from_text
        
        text = preset_file.read_text()
        text, _ = parse_envvars_from_text(text)
        if preset_file.suffix == ".json":
            data = json.loads(text)
        else:
            import yaml
            data = yaml.safe_load(text)
        
        if overrides: data.update(overrides)
        provider_settings = cls.parse_obj(data)
        ModelContextHandler.add_provider(provider_settings)
        return provider_settings
    
    @property
    def model_list(self) -> List[str]:
        """
        Returns the Model List (excludes the provider name)
        """
        names = [m.name for m in self.models]
        # Add aliases
        for m in self.models:
            if m.aliases:
                names.extend(m.aliases)
        return names


class ExternalProviderAuth(aiohttpx.Auth):
    """
    Custom Authentication Wrapper for External OpenAI Clients
    """
    def __init__(
        self, 
        config: ExternalProviderConfig,
        is_proxied: Optional[bool] = None,
        **kwargs,
    ):
        """
        Initializes the External Provider Auth

        :TODO - add support for Proxy
        """
        self.config = config
        self.is_proxied = is_proxied

    def auth_flow(self, request):
        """
        Injects the API Key into the Request
        """
        if self.config.has_api_key and self.config.api_key_header not in request.headers:
            request.headers[self.config.api_key_header] = self.config.api_key_value
        if self.config.custom_headers:
            request.headers.update(self.config.custom_headers)        
        yield request

    async def async_auth_flow(self, request):
        """
        Injects the API Key into the Request
        """
        if self.config.has_api_key and self.config.api_key_header not in request.headers:
            request.headers[self.config.api_key_header] = self.config.api_key_value
        if self.config.custom_headers:
            request.headers.update(self.config.custom_headers)
        yield request
        
        