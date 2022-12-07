from __future__ import absolute_import

from async_openai.utils import settings
from async_openai.api import OpenAI

# Allow for submodule calling
# following the old API behavior

Completions = OpenAI.completions
Edits = OpenAI.edits
