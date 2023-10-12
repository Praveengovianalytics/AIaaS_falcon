
# AIaaS Falcon

## Description

Generative AI - LLM library interacts with specific API, allowing operations such as listing models, creating embeddings, and generating text based on certain configurations.

## Installation

Ensure you have the `requests` and `google-api-core` libraries installed:

```bash
pip install aiaas-falcon
```

## Usage

1. **Initialization:**

   ```python
   from aiaas_falcon import Falcon # Assuming the class is saved in a file named falcon_client.py
   
   falcon = Falcon(api_key="<Your_API_Key>", host_name_port="<Your_Host_Name_Port>")
   ```

2. **Listing Models:**

   ```python
   models = falcon_client.list_models()
   print(models)
   ```

3. **Creating an Embedding:**

   ```python
   response = falcon_client.create_embedding(file_path="<Your_File_Path>")
   print(response)
   ```

4. **Generating Text:**

   ```python
   response = falcon_client.generate_text(chat_history=[], query="<Your_Query>")
   print(response)
   ```

## Methods

- `list_models(self)` - Retrieves available models.
- `create_embedding(self, file_path)` - Creates embeddings from a provided file.
- `generate_text(self, chat_history=[], query="", use_default=1, conversation_config={}, config={})` - Generates text based on provided parameters.

```python
# Example usage

from aiaas_falcon import Falcon  # Make sure the Falcon class is imported

# Initialize the Falcon object with the API key, host name and port
falcon = Falcon(api_key='_____API_KEY_____', host_name_port='34.16.138.59:8888', transport="rest")

# List available models
model = falcon.list_models()['models']

# Check if any model is available
if model:
    # Create an embedding
    response = falcon.create_embedding(['/content/01Aug2023.csv'])
    print(response)
    print('Embedding Success')

    # Define a prompt
    prompt = 'What is Account status key?'

    # Generate text based on the prompt and other parameters
    completion = falcon.generate_text(
         query=prompt,
         chat_history=[],
         use_default=1,
         conversation_config={
            "k": 5,
            "fetch_k": 50000,
            "bot_context_setting": "Do note that Your are a data dictionary bot. Your task is to fully answer the user's query based on the information provided to you."
         },
         config={"max_new_tokens": 1200, "temperature": 0.4, "top_k": 40, "top_p": 0.95, "batch_size": 256}
    )

    print(completion)
    print("Generate Success")

else:
    print("No suitable model found")
```


## Conclusion

The Falcon API Client simplifies interactions with the specified API, providing a straightforward way to perform various operations such as listing models, creating embeddings, and generating text.



## Authors

- [@Praveengovianalytics](https://github.com/Praveengovianalytics)
- [@zhuofan](https://github.com/zhuofan-16)


## Google Colab 
- [ Get start with aiaas_falcon ]] (https://colab.research.google.com/drive/1k5T_FO9SnlN0zOQfR7WFXSRFkfgiL1cE?usp=sharing)
## Badges

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
