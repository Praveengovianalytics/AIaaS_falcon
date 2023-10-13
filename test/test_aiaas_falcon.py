import time
from unittest.mock import Mock, patch

import pytest

from aiaas_falcon import \
    Falcon  # make sure to import your Falcon class correctly


# Define a fixture to get user inputs for api_key, host_name, and port
@pytest.fixture(scope="module")
def user_inputs():
    api_key = input("Enter API Key: ")
    host_name = input("Enter Hostname: ")
    port = input("Enter Port: ")
    return api_key, f"{host_name}:{port}"


def test_falcon_initialization(user_inputs):
    """
    Test the initialization of the Falcon class.

    This test checks whether the Falcon object is correctly initialized with
    the given parameters such as api_key and host_name_port.

    """
    api_key, host_name_port = user_inputs
    falcon = Falcon(api_key=api_key, host_name_port=host_name_port)
    assert falcon.api_key == api_key
    assert falcon.host_name_port == host_name_port


def test_list_models():
    """
    Test the list_models method of the Falcon class.

    This test checks whether the list_models method correctly returns a dictionary containing available models.
    """
    falcon = Falcon(api_key="test_api_key", host_name_port="localhost:8000")
    models = falcon.list_models()
    assert models == {"models": "llama2"}


@patch("requests.post")
def test_create_embedding(mock_post, user_inputs):
    """
    Test the create_embedding method of the Falcon class with mocked external API call.

    This test checks whether the create_embedding method correctly returns a successful status when the API response is mocked.
    """
    api_key, host_name_port = user_inputs
    mock_response = Mock()
    mock_response.json.return_value = {"status": "success"}
    mock_post.return_value = mock_response

    falcon = Falcon(api_key=api_key, host_name_port=host_name_port)
    response = falcon.create_embedding(file_path=["test_file_path"])

    assert response == {"status": "success"}


@patch("requests.post")
def test_generate_text(mock_post, user_inputs):
    """
    Test the generate_text method of the Falcon class with mocked external API call.

    This test checks whether the generate_text method correctly returns the expected generated text when the API response is mocked.
    """
    api_key, host_name_port = user_inputs
    mock_response = Mock()
    mock_response.json.return_value = {"generated_text": "Hello, world!"}
    mock_post.return_value = mock_response

    falcon = Falcon(api_key=api_key, host_name_port=host_name_port)
    response = falcon.generate_text(query="Hello?")

    assert response == {"generated_text": "Hello, world!"}


@patch("requests.post")
def test_create_embedding_time(mock_post, user_inputs):
    """
    Test the create_embedding method of the Falcon class with a time constraint.

    This test checks whether the create_embedding method completes within a certain time limit (e.g., 60 seconds).
    """
    api_key, host_name_port = user_inputs
    mock_response = Mock()
    mock_response.json.return_value = {"status": "success"}
    mock_post.return_value = mock_response

    falcon = Falcon(api_key=api_key, host_name_port=host_name_port)
    start_time = time.time()  # Record the start time before the request
    _ = falcon.create_embedding(file_path=["test_file_path"])
    # response = falcon.create_embedding(file_path=["test_file_path"])
    end_time = time.time()  # Record the end time after the request
    assert end_time - start_time < 60  # Ensure the request completes within 60 seconds


@patch("requests.post")
def test_generate_text_time(mock_post, user_inputs):
    """
    Test the generate_text method of the Falcon class with a time constraint.

    This test checks whether the generate_text method completes within a certain time limit (e.g., 10 seconds).
    """
    api_key, host_name_port = user_inputs
    mock_response = Mock()
    mock_response.json.return_value = {"generated_text": "Hello, world!"}
    mock_post.return_value = mock_response

    falcon = Falcon(api_key=api_key, host_name_port=host_name_port)
    start_time = time.time()  # Record the start time before the request
    _ = falcon.generate_text(query="Hello?")
    end_time = time.time()  # Record the end time after the request
    assert end_time - start_time < 10  # Ensure the request completes within 10 seconds
