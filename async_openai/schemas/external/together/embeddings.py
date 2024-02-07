from __future__ import annotations

"""
Together.xyz Embedding Route
"""


from ...embeddings import (
    EmbeddingRoute as BaseEmbeddingRoute,
    EmbeddingObject as BaseEmbeddingObject,
    EmbeddingResponse as BaseEmbeddingResponse,
    logger
)
from lazyops.types import validator, lazyproperty, Field
from async_openai.types.context import ModelContextHandler
from async_openai.types.resources import Usage
from typing import Any, Dict, List, Optional, Union, Set, Type, TYPE_CHECKING


class EmbeddingObject(BaseEmbeddingObject):
    model: Optional[str] = "togethercomputer/m2-bert-80M-32k-retrieval"

    @validator('model', pre=True, always=True)
    def validate_model(cls, v, values: Dict[str, Any]) -> str:
        """
        Validate the model
        """
        if not v:
            if values.get('engine'):
                v = values.get('engine')
            elif values.get('deployment'):
                v = values.get('deployment')
        v = ModelContextHandler.resolve_external_model_name(v)
        return v


class EmbeddingResponse(BaseEmbeddingResponse):

    usage: Optional[Usage] = Field(default_factory = Usage)


    @lazyproperty
    def consumption(self) -> int:
        """
        Returns the consumption for the completions
        """
        try:
            if not self.usage.prompt_tokens:
                self.usage.prompt_tokens = ModelContextHandler.count_tokens(self.input_object.input, model_name=self.input_object.model)
            return ModelContextHandler.get_consumption_cost(
                model_name = self.input_object.model,
                usage = self.usage,
            )
        except Exception as e:
            logger.error(f"Error getting consumption: {e}")
            return 0


class EmbeddingRoute(BaseEmbeddingRoute):
    input_model: Optional[Type[EmbeddingObject]] = EmbeddingObject
    response_model: Optional[Type[EmbeddingResponse]] = EmbeddingResponse