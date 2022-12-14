import aiohttpx

from typing import Optional, Dict, Callable
from async_openai.schemas import *
from async_openai.utils.config import settings

RouteClasses = {
    'completions': CompletionRoute,
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
    edits: EditRoute = None
    embeddings: EmbeddingRoute = None
    # files: FileRoute = None
    images: ImageRoute = None
    models: ModelRoute = None
    
    def __init__(
        self,
        client: aiohttpx.Client,
        headers: Optional[Dict] = None,
        debug_enabled: Optional[bool] = False,
        on_error: Optional[Callable] = None,
        ignore_errors: Optional[bool] = False,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        **kwargs
    ):
        self.client = client
        self.headers = headers or settings.get_headers()
        self.debug_enabled = debug_enabled
        self.on_error = on_error
        self.ignore_errors = ignore_errors
        self.timeout = timeout
        self.max_retries = max_retries
        self.kwargs = kwargs or {}
        self.init_routes()
    

    def init_routes(self):
        for route, route_class in RouteClasses.items():
            setattr(self, route, route_class(
                client = self.client,
                headers = self.headers,
                debug_enabled = self.debug_enabled,
                on_error = self.on_error,
                ignore_errors = self.ignore_errors,
                timeout = self.timeout,
                max_retries = self.max_retries,
                **self.kwargs
            ))
    
