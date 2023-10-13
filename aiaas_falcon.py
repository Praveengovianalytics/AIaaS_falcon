import requests
from google.api_core import retry

class Falcon:
    """
    Falcon class provides methods to interact with a specific API,
    allowing operations such as listing models, creating embeddings,
    and generating text based on certain configurations.
    """

    def __init__(self, api_key=None, host_name_port=None, transport=None):
        """
        Initialize the Falcon object with API key, host name and port, and transport.

        :param api_key: API key for authentication
        :param host_name_port: The host name and port where the API is running
        :param transport: Transport protocol (not currently used)
        """
        self.api_key = api_key  # API key for authentication
        self.host_name_port = host_name_port  # host and port information
        self.transport = transport  # transport protocol (not used)
        self.headers = {
            "Authorization": api_key,
        }  # headers for authentication

    def list_models(self):
        """
        List the available models from the API.

        :return: A dictionary containing available models.
        """
        return {"models": "llama2"}

    def create_embedding(self, file_path):
        """
        Create embeddings by sending files to the API.

        :param file_path: Paths of the files to be uploaded
        :return: JSON response from the API
        """
        url = f"http://{self.host_name_port}/v1/chat/create_embeddingLB"

        # Opening files in read mode
        files = [("file", open(item, "r")) for item in file_path]

        # Preparing data with file extensions
        data = {"extension": ["".join(item.split(".")[-1]) for item in file_path]}

        headers = {
            "X-API-Key": self.api_key,
        }  # headers with API key

        # Making a POST request to the API
        response = requests.post(url, headers=headers, files=files, data=data)
        response.raise_for_status()  # raising exception for HTTP errors
        return response.json()  # returning JSON response

    @retry.Retry()
    def generate_text(
        self,
        chat_history=[],
        query="",
        use_default=1,
        conversation_config={
            "k": 2,
            "fetch_k": 50,
            "bot_context_setting": "Do note that You are a data dictionary bot. Your task is to fully answer the user's query based on the information provided to you.",
        },
        config={
            "max_new_tokens": 1200,
            "temperature": 0.4,
            "top_k": 40,
            "top_p": 0.95,
            "batch_size": 256,
        },
    ):
        """
        Generate text by sending data to the API.

        :param chat_history: Chat history for context
        :param query: Query to be asked
        :param use_default: Flag to use default configuration
        :param conversation_config: Conversation configuration parameters
        :param config: Other configuration parameters
        :return: JSON response from the API
        """
        url = f"http://{self.host_name_port}/v1/chat/predictLB"

        # Preparing data to be sent in the request
        data = {
            "chat_history": chat_history,
            "query": query,
            "use_default": use_default,
            "conversation_config": conversation_config,
            "config": config,
        }

        headers = {
            "X-API-Key": self.api_key,
        }  # headers with API key

        # Making a POST request to the API
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # raising exception for HTTP errors
        return response.json()  # returning JSON response
