import datetime
from enum import Enum

from pydantic import validator
from typing import List, Any, Optional, Union, Dict, Type
from async_openai.types import BaseModel, lazyproperty
from async_openai.schemas.types.base import BaseResult, EventResult, Method, BaseEndpoint
from async_openai.schemas.types.files import FileRequest, FileData
from async_openai.utils import logger

__all__ = [
    'FinetuneModels',
    'FinetuneRequest',
    'FinetuneData',
    'FinetuneResult',
]

class FinetuneModels(str, Enum):
    """
    Supported finetune models.
    """
    ada = "ada"
    babbage = "babbage"
    curie = "curie"
    davici = "davici"


class FinetuneRequest(BaseModel):
    """Finetune request model"""

    fine_tune_id: Optional[str]
    training_file: Optional[str] # file-xxxx
    validation_file: Optional[str] # file-xxxx

    model: Optional[Union[str, FinetuneModels]] = FinetuneModels.curie
    n_epochs: Optional[int] = 4
    batch_size: Optional[int] = None
    learning_rate_multiplier: Optional[float] = None
    prompt_loss_weight: Optional[float] = 0.01
    compute_classification_metrics: Optional[bool] = False
    classification_n_classes: Optional[int] = None
    classification_betas: Optional[List[float]] = None
    suffix: Optional[str] = None
    stream: bool = False

    @validator("batch_size")
    def validate_batch_size(cls, value):
        """
        Batch Size is capped at 256
        """
        return value if value is None else min(int(value), 256)

    @validator("learning_rate_multiplier")
    def validate_learning_rate_multiplier(cls, value):
        """
        The learning rate multiplier to use for training. 
        The fine-tuning learning rate is the original learning rate used for 
        pretraining multiplied by this value.

        By default, the learning rate multiplier is the 0.05, 0.1, or 0.2 
        depending on final batch_size 
        (larger learning rates tend to perform better with larger batch sizes). 
        We recommend experimenting with values in the range 0.02 to 0.2 to see 
        what produces the best results.
        """
        return value if value is None else min(float(value), 2.0)
    
    @validator("model")
    def validate_model(cls, value):
        """
        Supported models:
        - ada
        - babbage
        - curie
        - davici
        """
        if value is None:
            return FinetuneModels.curie
        if isinstance(value, str) and value in FinetuneModels.__members__:
            return FinetuneModels(value)
        logger.warning(f'{value} is not a validated model. Skipping check.')
        return value
    
    @lazyproperty
    def create_finetune_endpoint(self) -> BaseEndpoint:
        return BaseEndpoint(
            method = Method.POST,
            url = '/fine-tunes',
            data = self.dict(
                exclude_none = True,
                exclude = {'stream'},
            )
        )
    
    @lazyproperty
    def list_finetune_endpoint(self) -> BaseEndpoint:
        return BaseEndpoint(
            method = Method.GET,
            url = '/fine-tunes'
        )

    @property
    def retrieve_finetune_endpoint(self) -> BaseEndpoint:
        return BaseEndpoint(
            method = Method.GET,
            url = f'/fine-tunes/{self.fine_tune_id}'
        )
    
    @property
    def cancel_finetune_endpoint(self) -> BaseEndpoint:
        return BaseEndpoint(
            method = Method.POST,
            url = f'/fine-tunes/{self.fine_tune_id}/cancel'
        )
    
    @property
    def list_finetune_events_endpoint(self) -> BaseEndpoint:
        return BaseEndpoint(
            method = Method.GET,
            url = f'/fine-tunes/{self.fine_tune_id}/events',
            params = {
                'stream': self.stream,
            }
        )
    
    @property
    def delete_finetune_model_endpoint(self) -> BaseEndpoint:
        return BaseEndpoint(
            method = Method.DELETE,
            url = f'/models/{self.model}'
        )

    
class FinetuneData(BaseResult):
    id: Optional[str]
    object: Optional[str] = 'fine-tune'
    model: Optional[str]
    created_at: Optional[datetime.datetime]
    fine_tuned_model: Optional[str]
    hyperparams: Optional[Dict[str, Any]]
    organization_id: Optional[str]
    result_files: Optional[List[FileData]]
    status: Optional[str]
    training_files: Optional[List[FileData]]
    validation_files: Optional[List[FileData]]
    updated_at: Optional[datetime.datetime]

    @validator("created_at")
    def validate_created_at(cls, value):
        return datetime.datetime.fromtimestamp(value, datetime.timezone.utc) if value else value
    
    @validator("updated_at")
    def validate_updated_at(cls, value):
        return datetime.datetime.fromtimestamp(value, datetime.timezone.utc) if value else value
    


class FinetuneResult(BaseResult):
    events: Optional[List[EventResult]]
    data: Optional[Union[FinetuneData, List[FinetuneData]]]

    _event_model: Optional[Type[EventResult]] = EventResult
    _data_model: Optional[Type[FinetuneData]] = FinetuneData

    _request: Optional[FinetuneRequest] = None

    fine_tuned_model: Optional[str]
    hyperparams: Optional[Dict[str, Any]]
    organization_id: Optional[str]
    result_files: Optional[List[FileData]]
    training_files: Optional[List[FileData]]
    validation_files: Optional[List[FileData]]
    
    status: Optional[str]
    updated_at: Optional[datetime.datetime]

    @property
    def metadata_fields(self):
        return [
            'id',
            'object',
            'created_at',
            'fine_tuned_model',
            'hyperparams',
            'organization_id',
            'result_files',
            'training_files',
            'validation_files',
            'status',
            'updated_at',
            # 'events',
            # 'data',
        ]

