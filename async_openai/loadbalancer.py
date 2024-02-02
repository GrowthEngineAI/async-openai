"""
Client LoadBalancer
"""

from __future__ import annotations

import random
from typing import Optional, List, Dict, Union, TYPE_CHECKING

from async_openai.schemas import *
from async_openai.utils.config import get_settings, OpenAISettings
from async_openai.utils.logs import logger

if TYPE_CHECKING:
    from async_openai.client import OpenAIClient, OpenAISessionManager


class ClientLoadBalancer:
    """
    Manages a set of clients that can be rotated.    
    """
    def __init__(
        self, 
        prioritize: Optional[str] = None, 
        settings: Optional[OpenAISettings] = None, 
        azure_model_mapping: Optional[Dict[str, str]] = None, 
        healthcheck: Optional[bool] = True,
        manager: Optional['OpenAISessionManager'] = None,
    ):
        self.settings = settings or get_settings()
        self.clients: Dict[str, 'OpenAIClient'] = {}
        self.rotate_index: int = 0
        self.rotate_client_names: List[str] = []
        self.azure_model_mapping: Dict[str, str] = azure_model_mapping
        self.healthcheck: bool = healthcheck
        self.manager: Optional['OpenAISessionManager'] = manager

        assert prioritize in [None, 'azure', 'openai'], f'Invalid `prioritize` value: {prioritize}'
        self.prioritize: Optional[str] = prioritize
    
    @property
    def client_names(self) -> List[str]:
        """
        Returns the list of client names.
        """
        return list(self.clients.keys())
    
    def run_client_init(self):
        """
        Initializes the Client. 

        Can be subclassed to provide custom initialization.
        """
        self.init_api_client()
        if self.settings.has_valid_azure:
            self.init_api_client(client_name = 'az', is_azure = True, set_as_default = self.prioritize == 'azure', set_as_current = self.prioritize == 'azure')


    @property
    def api(self) -> 'OpenAIClient':
        """
        Returns the inherited OpenAI client.
        """
        if not self.clients: 
            self.run_client_init()
        if not self.rotate_client_names or self.rotate_index < len(self.client_names):
            return self.clients[self.client_names[self.rotate_index]]
        try:
            return self.clients[self.rotate_client_names[self.rotate_index]]
        except IndexError as e:
            logger.error(f'Index Error: {self.rotate_index} - {self.rotate_client_names}')
            raise IndexError(f'Index Error: {self.rotate_index} - {self.rotate_client_names} - {self.client_names} ({len(self.clients)})') from e
    
    def increase_rotate_index(self):
        """
        Increases the rotate index
        """
        if self.rotate_index >= len(self.clients) - 1:
            self.rotate_index = 0
        else:
            self.rotate_index += 1

    def rotate_client(self, index: Optional[int] = None, require_azure: Optional[bool] = None, verbose: Optional[bool] = False):
        """
        Rotates the clients
        """
        if index is not None:
            self.rotate_index = index
            return
        self.increase_rotate_index()
        if require_azure:
            while not self.api.is_azure:
                self.increase_rotate_index()
        if verbose:
            logger.info(f'Rotated Client: {self.api.name} (Azure: {self.api.is_azure} - {self.api.api_version}) [{self.rotate_index+1}/{len(self.clients)}]')
    
    def set_client(self, client_name: Optional[str] = None, verbose: Optional[bool] = False):
        """
        Sets the client
        """
        if client_name is None:
            raise ValueError('`client_name` is required.')
        if client_name not in self.clients:
            raise ValueError(f'Client `{client_name}` does not exist.')
        self.rotate_index = self.client_names.index(client_name)
        if verbose:
            logger.info(f'Set Client: {self.api.name} (Azure: {self.api.is_azure} - {self.api.api_version})) [{self.rotate_index+1}/{len(self.clients)}]')

    def current_client_info(self, verbose: Optional[bool] = False) -> Dict[str, Union[str, int]]:
        """
        Returns the current client info
        """
        data = {
            'name': self.api.name,
            'is_azure': self.api.is_azure,
            'api_version': self.api.api_version,
            'index': self.rotate_index,
            'total': len(self.clients),
        }
        if verbose:
            logger.info(f'Current Client: {self.api.name} (Azure: {self.api.is_azure} - {self.api.api_version}) [{self.rotate_index+1}/{len(self.clients)}]')
        return data


    def configure_client(self, client_name: Optional[str] = None, priority: Optional[int] = None, **kwargs):
        """
        Configure a new client
        """
        client_name = client_name or 'default'
        if client_name not in self.clients:
            raise ValueError(f'Client `{client_name}` does not exist.')
        self.clients[client_name].reset(**kwargs)
        if priority is not None:
            if client_name in self.rotate_client_names:
                self.rotate_client_names.remove(client_name)
            self.rotate_client_names.insert(priority, client_name)

    def init_api_client(
        self, 
        client_name: Optional[str] = None, 
        set_as_default: Optional[bool] = False, 
        is_azure: Optional[bool] = None,
        priority: Optional[int] = None,
        set_as_current: Optional[bool] = False,
        **kwargs
    ) -> 'OpenAIClient':
        """
        Creates a new OpenAI client.
        """
        client_name = client_name or 'default'
        if client_name in self.clients:
            return self.clients[client_name]

        from async_openai.client import OpenAIClient
        if is_azure is None and \
                (
                'az' in client_name and self.settings.has_valid_azure
            ):
            is_azure = True
        if 'client_callbacks' not in kwargs and \
                self.manager and \
                self.manager.client_callbacks:
            kwargs['client_callbacks'] = self.manager.client_callbacks
        client = OpenAIClient(
            name = client_name,
            settings = self.settings,
            is_azure = is_azure,
            azure_model_mapping = self.azure_model_mapping,
            **kwargs
        )
        self.clients[client_name] = client
        if set_as_default:
            self.rotate_client_names.insert(0, client_name)
        elif priority is not None:
            if client_name in self.rotate_client_names:
                self.rotate_client_names.remove(client_name)
            self.rotate_client_names.insert(priority, client_name)
        elif self.prioritize:
            if (
                self.prioritize == 'azure'
                and is_azure
                or self.prioritize != 'azure'
                and self.prioritize == 'openai'
                and not is_azure
            ):
                self.rotate_client_names.insert(0, client_name)
            elif self.prioritize in ['azure', 'openai']:
                self.rotate_client_names.append(client_name)
        if set_as_current:
            self.rotate_index = self.rotate_client_names.index(client_name)
        return client
    
    def get_api_client(self, client_name: Optional[str] = None, require_azure: Optional[bool] = None, **kwargs) -> 'OpenAIClient':
        """
        Initializes a new OpenAI client or Returns an existing one.
        """
        if not client_name and not self.clients:
            client_name = 'default'
        if client_name and client_name not in self.clients:
            self.clients[client_name] = self.init_api_client(client_name = client_name, **kwargs)

        if not client_name and require_azure:
            while not self.api.is_azure:
                self.increase_rotate_index()
            return self.api
        return self.clients[client_name] if client_name else self.api
    
    def get_api_client_from_list(self, client_names: List[str], require_azure: Optional[bool] = None, **kwargs) -> 'OpenAIClient':
        """
        Initializes a new OpenAI client or Returns an existing one from a list of client names.
        """
        if not self.healthcheck:
            name = random.choice(client_names)
            return self.get_api_client(client_name = name, require_azure = require_azure, **kwargs)
        for client_name in client_names:
            if client_name not in self.clients:
                self.clients[client_name] = self.init_api_client(client_name = client_name, **kwargs)
            if require_azure and not self.clients[client_name].is_azure:
                continue
            if not self.clients[client_name].ping():
                continue
            return self.clients[client_name]
        raise ValueError(f'No healthy client found from: {client_names}')
    
    async def aget_api_client_from_list(self, client_names: List[str], require_azure: Optional[bool] = None, **kwargs) -> 'OpenAIClient':
        """
        Initializes a new OpenAI client or Returns an existing one from a list of client names.
        """
        if not self.healthcheck:
            name = random.choice(client_names)
            return self.get_api_client(client_name = name, require_azure = require_azure, **kwargs)
        for client_name in client_names:
            if client_name not in self.clients:
                self.clients[client_name] = self.init_api_client(client_name = client_name, **kwargs)
            if require_azure and not self.clients[client_name].is_azure:
                continue
            if not await self.clients[client_name].aping():
                continue
            return self.clients[client_name]
        raise ValueError(f'No healthy client found from: {client_names}')

    def __getitem__(self, key: Union[str, int]) -> 'OpenAIClient':
        """
        Returns a client by name.
        """
        if isinstance(key, int):
            key = self.rotate_client_names[key] if self.rotate_client_names else self.client_names[key]
        return self.clients[key]