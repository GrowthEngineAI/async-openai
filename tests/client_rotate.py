from async_openai import OpenAI

org_id = 'org-...'
api_key = 'sk-...'

azure_api_base = "https://....openai.azure.com/"

# azure_api_version = "2023-03-15-preview"
azure_api_version = "2023-07-01-preview"
azure_api_key = "...."

OpenAI.configure(
    # OpenAI Configuration
    api_key = api_key,
    organization = org_id,
    debug_enabled = True,

    # Azure Configuration
    azure_api_base = azure_api_base,
    azure_api_version = azure_api_version,
    azure_api_key = azure_api_key,
    enable_rotating_clients = True,
    prioritize = "azure",
)


print(OpenAI.settings.azure.dict())
OpenAI.get_current_client_info(verbose = True)
OpenAI.rotate_client(verbose = True)