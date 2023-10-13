from aiaas_falcon import Falcon

# assume the library is installed and available as aiaas_falcon

falcon = Falcon(
    api_key="TESTKEY123", host_name_port="34.16.138.59:8888", transport="rest"
)
model = falcon.list_models()["models"]

if model:
    response = falcon.create_embedding(["/Users/praveen/downloads/01Aug2023.csv"])
    print(response)
    print("Embedding Success")

    prompt = "What is Account status key?"
    completion = falcon.generate_text(
        query=prompt,
        chat_history=[],
        use_default=1,
        conversation_config={
            "k": 5,
            "fetch_k": 50000,
            "bot_context_setting": "Do note that Your are a data dictionary bot. Your task is to fully answer the user's query based on the information provided to you.",
        },
        config={
            "max_new_tokens": 1200,
            "temperature": 0.4,
            "top_k": 40,
            "top_p": 0.95,
            "batch_size": 256,
        },
    )

    print(completion)
    print("Generate Success")

else:
    print("No suitable model found")
