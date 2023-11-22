import requests
from google.api_core import retry
import numpy as np
import requests

class Falcon:
    """
    Falcon class provides methods to interact with a specific API,
    allowing operations such as listing models, creating embeddings,
    and generating text based on certain configurations.
    """

    def __init__(self, api_key=None, host_name_port=None, api_type='',transport=None,protocol='http'):
        """
        Initialize the Falcon object with API key, host name and port, and transport.

        :param api_key: API key for authentication
        :param host_name_port: The host name and port where the API is running
        :param transport: Transport protocol (not currently used)
        """
        self.api_key = api_key  # API key for authentication
        api_type=f'/{api_type}' if api_type else ''
        self.host_name_port = host_name_port+f'{api_type}'  # host and port information
        self.transport = transport  # transport protocol (not used)
        self.protocol=protocol
        self.headers = {
            "Authorization": api_key,
        }  # headers for authentication

    def list_models(self):
        """
        List the available models from the API.

        :return: A dictionary containing available models.
        """
        url = f"{self.protocol}://{self.host_name_port}/v1/chat/get_model"

        response = requests.get(url,verify=False)

        return response.json()

    def health(self):
        """
        List the available models from the API.

        :return: A dictionary containing available models.
        """
        url = f"{self.protocol}://{self.host_name_port}/v1/chat/ping"
        response = requests.get(url)

        return response.json()

    def create_embedding(self, file_path, type='general'):
        """
        Create embeddings by sending files to the API.

        :param file_path: Paths of the files to be uploaded
        :return: JSON response from the API
        """
        url = f"{self.protocol}://{self.host_name_port}/v1/chat/create_embeddingLB"

        # Opening files in read mode
        files = [("file", open(item, "r")) for item in file_path]

        # Preparing data with file extensions
        data = {"extension": ["".join(item.split(".")[-1]) for item in file_path], "type": type}

        headers = {
            "X-API-Key": self.api_key,
        }  # headers with API key

        # Making a POST request to the API
        response = requests.post(url, headers=headers, verify=False,files=files, data=data)
        response.raise_for_status()  # raising exception for HTTP errors
        return response.json()  # returning JSON response

    @retry.Retry()
    def generate_text(
            self,
            query=""
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
        url = f"{self.protocol}://{self.host_name_port}/v1/chat/predictLB"

        conversation_config={
                "k": 8,
                "fetch_k": 100000,
                "bot_context_setting":"" ,
            }

        use_file=0

        chat_history=[]

        use_default=1

        type='general'

        config={
                "model": 'llama2-13b',
                "max_new_tokens": 4000,
                "temperature": 0,
                "top_p": 0.95,
                "batch_size": 256
            }

        # Preparing data to be sent in the request
        data = {
            "chat_history": chat_history,
            "query": query,
            "use_default": use_default,
            'use_file': use_file,
            "conversation_config": conversation_config,
            "config": config,
            'type': type
        }

        headers = {
            "X-API-Key": self.api_key,
        }  # headers with API key

        # Making a POST request to the API
        response = requests.post(url, headers=headers, verify=False,json=data)
        response.raise_for_status()  # raising exception for HTTP errors
        return response.json()  # returning JSON response




    @retry.Retry()
    def generate_text_full(
            self,
            query:str="",
            max_new_tokens:int=4000,
            temperature:float=0,
            top_k:int=-1,
    ):
        """_summary_

        Args:
            query (str, optional): _description_. Defaults to "".
            max_new_tokens (int, optional): _description_. Defaults to 4000 because llama2-13B model used.
            temperature (float, optional): _description_. Defaults to 0.
            top_k (int, optional): _description_. Defaults to -1.

        Returns:
         //   [type]: JSON respose from the API Status:str message:list

        """
        url = f"{self.protocol}://{self.host_name_port}/v1/chat/predict-CCT"

        # Preparing data to be sent in the request
        data = {
            "query": query,
            "temperature":temperature,
            "top_k":top_k,
            "max_tokens":max_new_tokens
        }

        headers = {
            "X-API-Key": self.api_key,
        }  # headers with API key

        # Making a POST request to the API
        response = requests.post(url, headers=headers, verify=False,json=data)
        response.raise_for_status()  # raising exception for HTTP errors
        return response.json()  # returning JSON response






    @retry.Retry()
    def generate_text_lah(
            self,
            query="",
            context="",
            config={
                "model": 'llama2-13b',
                "max_new_tokens": 4000,
                "temperature": 0,
                "top_p": 0.95,
                "batch_size": 256
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
        url = f"{self.protocol}://{self.host_name_port}/v1/chat/predictLB"

        # Preparing data to be sent in the request
        type='general'

        data = {
            "chat_history": [],
            "query": query,
            "use_default": 1,
            'use_file': 0,
            "conversation_config": {"k": 8,
                "fetch_k": 100000,
                "bot_context_setting":context},
            "config": config,
            'type': type
        }

        headers = {
            "X-API-Key": self.api_key,
        }  # headers with API key

        # Making a POST request to the API
        response = requests.post(url, headers=headers, verify=False,json=data)
        response.raise_for_status()  # raising exception for HTTP errors
        return response.json()  # returning JSON response


class FalconAudio:
    def __init__(self, host_name_port, protocol, api_key,api_type='transcribe', transport=None):
        self.api_url = f"{protocol}://{host_name_port}/{api_type}/audio_2_text"
        self.api_key = api_key
        self.headers = {
            "X-API-Key": self.api_key
        }

    def transcribe(self, audio_data:list,sampling_rate=16000):
        payload = {
            "audio": audio_data,
            "sampling_rate": sampling_rate
        }

        response = requests.post(self.api_url, json=payload, headers=self.headers)
        if response.status_code == 200:
            return response.json()
        else:
            return response.text
