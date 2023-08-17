import aiohttpx

from typing import Optional, Dict, Callable, TYPE_CHECKING
from async_openai.schemas import *
from async_openai.utils.config import get_settings, OpenAISettings, AzureOpenAISettings
from async_openai.utils.logs import logger


RouteClasses = {
    'completions': CompletionRoute,
    'chat': ChatRoute,
    'edits': EditRoute,
    'embeddings': EmbeddingRoute,
    # 'files': FileRoute,
    'images': ImageRoute,
    'models': ModelRoute,

}

class ApiRoutes:

    """
    Container for all the routes in the API.
    """

    completions: CompletionRoute = None
    chat: ChatRoute = None
    edits: EditRoute = None
    embeddings: EmbeddingRoute = None
    # files: FileRoute = None
    images: ImageRoute = None
    models: ModelRoute = None
    
    def __init__(
        self,
        client: aiohttpx.Client,
        # headers: Optional[Dict] = None,
        debug_enabled: Optional[bool] = False,
        on_error: Optional[Callable] = None,
        ignore_errors: Optional[bool] = False,
        disable_retries: Optional[bool] = None,
        retry_function: Optional[Callable] = None,

        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        settings: Optional[OpenAISettings] = None,
        is_azure: Optional[bool] = None,
        
        **kwargs
    ):
        self.client = client
        self.settings = settings or get_settings()
        # self.headers = headers or self.settings.get_headers()
        self.debug_enabled = debug_enabled
        self.on_error = on_error
        self.ignore_errors = ignore_errors
        self.disable_retries = disable_retries
        self.retry_function = retry_function

        self.timeout = timeout
        self.max_retries = max_retries
        self.is_azure = is_azure if is_azure is not None else \
            isinstance(self.settings, AzureOpenAISettings)
        self.kwargs = kwargs or {}
        self.init_routes()
    


    def init_routes(self):
        """
        Initializes the routes
        """
        for route, route_class in RouteClasses.items():
            try:
                setattr(self, route, route_class(
                    client = self.client,
                    # headers = self.headers,
                    debug_enabled = self.debug_enabled,
                    on_error = self.on_error,
                    ignore_errors = self.ignore_errors,
                    disable_retries = self.disable_retries,
                    retry_function = self.retry_function,
                    timeout = self.timeout,
                    max_retries = self.max_retries,
                    settings = self.settings,
                    is_azure = self.is_azure,
                    **self.kwargs
                ))
            except Exception as e:
                logger.error(f"Failed to initialize route {route} with error: {e}")
                raise e
