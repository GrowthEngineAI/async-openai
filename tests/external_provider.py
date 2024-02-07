import os

os.environ['TOGETHER_API_KEY'] = 'test123'

from async_openai.utils.external_config import ExternalProviderSettings

def test_external_provider():
    s = ExternalProviderSettings.from_preset('together')
    print(s)

test_external_provider()