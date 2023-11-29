
import random

from google.api_core import retry
import numpy as np
import requests
from aiaas_falcon_light.aiaas_falcon_light import Light
class Falcon:
    """
    Falcon class provides methods to interact with a specific API,
    allowing operations such as listing models, creating embeddings,
    and generating text based on certain configurations.
    """

    def __init__(self, api_key=None, api_name=None, api_endpoint='dev_quan', host_name_port='', api_type='',
                 transport=None, protocol='http',use_pil=False,log_key=None):
        """
        Initialize the Falcon object with API key, host name and port, and transport.

        :param api_key: API key for authentication
        :param host_name_port: The host name and port where the API is running
        :param transport: Transport protocol (not currently used)
        """
        self.endpoint = []
        self.log_id=random.randint(1000000, 9999999)
        api_type = f'/{api_type}' if api_type else ''
        self.host_name_port = host_name_port + f'{api_type}'  # host and port information
        self.transport = transport  # transport protocol (not used)
        self.protocol = protocol
        api_name = api_name if api_name else 'default'
        if api_endpoint!='prod' and api_endpoint!='dev_full' and api_endpoint!='dev_quan' and api_endpoint!='azure':
            raise Exception("Invalid API Type")
        if api_endpoint=='dev_full' or api_endpoint=='dev_quan':
            headers={
                "X-API-Key": api_key,
            }
        elif api_endpoint=='prod':
            headers={"X-API-Key": api_key,
}
        else:
            headers={
                "api-key": api_key
            }

        self.endpoint.append(
            {'name': api_name, 'url': f"{self.protocol}://{self.host_name_port}", 'type': api_endpoint, 'auth': api_key,
             'headers': headers ,'log_id':self.log_id,"use_pil":use_pil,"log_key":log_key})

        self.current = 0
        self.falcon = Light(self.endpoint[self.current])


    def current_active(self):
        return self.endpoint[self.current]
    def add_endpoint(self, api_name,protocol,host_name_port,api_endpoint,api_key,use_pil=False,log_key=None):
        self.log_id=random.randint(1000000, 9999999)
        if api_endpoint!='prod' and api_endpoint!='dev_full' and api_endpoint!='dev_quan' and api_endpoint!='azure':
            raise Exception("Invalid API Type")
        if api_endpoint == 'dev_full'or api_endpoint=='dev_quan':
            headers = {
                "X-API-Key": api_key,
            }
        elif api_endpoint == 'prod':
            headers = {"X-API-Key": api_key,
                       }
        else:
            headers = {
                "api-key": api_key
            }

        self.endpoint.append(
            {'name': api_name, 'url': f"{protocol}://{host_name_port}", 'type': api_endpoint, 'auth': api_key,
             'headers': headers,'log_id':self.log_id,"use_pil":use_pil,"log_key":log_key})
        return "Add Success"

    def list_endpoint(self):
        return self.endpoint

    def set_endpoint(self,name):
        self.current = next((index for index, obj in enumerate(self.endpoint) if obj['name'] == name), None)
        if not self.current:
            return "Invalid endpoint"
        self.falcon = Light(self.endpoint[self.current])

        return f"Set {name} success"
    def remove_endpoint(self,name):
        index=next((index for index, obj in enumerate(self.endpoint) if obj['name'] == name), None)
        if index:
            self.endpoint.pop(index)
        else:
            pass
        return "Remove Success"



    def list_models(self):
        """
        List the available models from the API.

        :return: A dictionary containing available models.
        """
        return self.falcon.list_models()
    def health(self):
        """
        List the available models from the API.

        :return: A dictionary containing available models.
        """

        return self.falcon.health()

    def create_embedding(self, file_path, type='general'):
        """
        Create embeddings by sending files to the API.

        :param file_path: Paths of the files to be uploaded
        :return: JSON response from the API
        """
        return self.falcon.create_embedding(file_path)  # returning JSON response

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

    def current_pil(self):
        return self.falcon.current_pil()
    def switch_pil(self):
        return self.falcon.switch_pil()
    def initialise_pil(self):
        return self.falcon.initialise_pil()

    def evaluate_parameter(self,config):
        return self.falcon.evaluate_parameter(config)

    def decode_config(self,text):
        return self.falcon.decrypt_hash(text)
    @retry.Retry()
    def generate_text_full(
            self,
            query="",
            context="",
            model="",
            use_file=0,
            chat_history=[],
            max_new_tokens: int = 4000,
            temperature: float = 0,
            top_k: int = -1,
            frequency_penalty: int = 1,
            repetition_penalty: int = 1,
            presence_penalty: float = 1.5,
            fetch_k=100000,
            select_k=4,
            api_version='2023-05-15',
            guardrail={'jailbreak':False,'moderation':False},
            custom_guardrail=None
    ):
        """_summary_

        Args:
            query (str, optional): _description_. Defaults to "".
            max_new_tokens (int, optional): _description_. Defaults to 4000 because llama2-13B model used.
            temperature (float, optional): _description_. Defaults to 0.
            top_k (int, optional): _description_. Defaults to -1.
            frequency_penalty:int=1,
            repetition_penalty:int=1,
            presence_penalty:float=1.5

        Returns:
         //   [type]: JSON respose from the API Status:str message:list

        """
        return self.falcon.generate_text(
            model=model,
               query=query,
               context=context,
               use_file=use_file,
               chat_history=chat_history,
               max_new_tokens=max_new_tokens,
               temperature=temperature,
               top_k=top_k,
               frequency_penalty=frequency_penalty,
               repetition_penalty=repetition_penalty,
               presence_penalty=presence_penalty,
               fetch_k=fetch_k,
               select_k=select_k,
               api_version=api_version,
            guardrail=guardrail,
            custom_guardrail=custom_guardrail
           )

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
