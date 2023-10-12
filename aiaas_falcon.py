import requests
from google.api_core import retry

class Falcon:

    def __init__(self, api_key=None, host_name_port=None, transport=None):

        self.api_key = api_key
        self.host_name_port = host_name_port
        self.transport = transport
        self.headers = {'Authorization': api_key,}

    def list_models(self):
      return {"models":'llama2'}

    def create_embedding(self, file_path):
        url = f'http://{self.host_name_port}/v1/chat/create_embeddingLB'

        # Assuming 'file_path' is the local file path to the file you want to upload
        files = [('file', open(item, 'r')) for item in file_path]
        data = {'extension': ["".join(item.split('.')[-1])  for item in file_path]}

        headers = {
            'X-API-Key': self.api_key,
        }

        response = requests.post(url, headers=headers, files=files, data=data)
        response.raise_for_status()
        return response.json()
    @retry.Retry()
    def generate_text(self, chat_history=[], query="", use_default=1, conversation_config={
            "k":2,
            "fetch_k":50,
            "bot_context_setting":"Do note that Your are a data dictionary bot. Your task is to fully answer the user's query based on the information provided to you."
    }, config={"max_new_tokens": 1200, "temperature": 0.4, "top_k": 40, "top_p": 0.95, "batch_size": 256}):
        url = f'http://{self.host_name_port}/v1/chat/predictLB'

        data = {
            "chat_history": chat_history,
            "query": query,
            "use_default": use_default,
            "conversation_config": conversation_config,
            "config": config
        }

        headers = {
            'X-API-Key': self.api_key,
        }

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()