from abc import ABC, abstractmethod
from typing import Dict, Generator, List, Union
from urllib.parse import urlparse

import requests
from azure.ai.ml import MLClient
from pydantic import BaseModel

from src.utils import log_message, show_ml_info


class AISimpleResponse(BaseModel):
    """Represent a response item of respond() func

    Args:
        bot_message (str): API Response Message
        chat_history (List[Dict[str, str]]): List of ChatHistory Json for AOAI.
            The Format is `[ {"role": "user", "metadata": {"title": None}, "content": "<USER MESSAGE1>"}, {"role": "assistant", "metadata": {"title": None}, "content": "<RESPONSE MESSAGE1>"},
                {"role": "user", "metadata": {"title": None}, "content": "<USER MESSAGE2>"}, {"role": "assistant", "metadata": {"title": None}, "content": "<RESPONSE MESSAGE2>"} ]`

        chat_history_for_ml (List[Dict[str, str | Dict]]): LIST of ChatHistory Json for MLAPI.
            The Format is `[ {"inputs": {"question": "<USER MESSAGE1>"}, "outputs": "<RESPONSE MESSAGE1>"},
                {"inputs": {"question": "<USER MESSAGE2>"}, "outputs": "<RESPONSE MESSAGE2>"},・・・]`
    """

    bot_message: str
    chat_history: List[Dict[str, str | Dict]]
    chat_history_for_ml: List[Dict[str, str | Dict]]


class AICustomResponse(BaseModel):
    """Represent a response item of respond() func

    Args:
        bot_message (str): API Response Message
        chat_history (List[Dict[str, str]]): List of ChatHistory Json for AOAI.
            The Format is `[ {"role": "user", "metadata": {"title": None}, "content": "<USER MESSAGE1>"}, {"role": "assistant", "metadata": {"title": None}, "content": "<RESPONSE MESSAGE1>"},
                {"role": "user", "metadata": {"title": None}, "content": "<USER MESSAGE2>"}, {"role": "assistant", "metadata": {"title": None}, "content": "<RESPONSE MESSAGE2>"} ]`

        chat_history_for_ml (List[Dict[str, str | Dict]]): LIST of ChatHistory Json for MLAPI.
            The Format is `[ {"inputs": {"question": "<USER MESSAGE1>"}, "outputs": "<RESPONSE MESSAGE1>"},
                {"inputs": {"question": "<USER MESSAGE2>"}, "outputs": "<RESPONSE MESSAGE2>"},・・・]`

        call_history (str): Past call log HTML.
        call_log_md_display (str): Completed Call log HTML.
        call_count (int): API call count.
    """

    bot_message: str
    chat_history: List[Dict[str, str | Dict]]
    chat_history_for_ml: List[Dict[str, str | Dict]]

    call_history: str
    call_log_md_display: str
    call_count: int


class BaseChatApp(ABC):
    def __init__(
        self,
        ml_client: MLClient,
        endpoint_name: str,
        deployment_name: str,
    ) -> None:
        """Initializes the ChatApp with Azure Machine Learning client and endpoint information.

        Args:
            ml_client (MLClient): The Azure Machine Learning client.
            endpoint_name (str): Name of the online endpoint.
            deployment_name (str): Name of the deployment in the endpoint.
        """

        log_message("Getting endpoint info...")
        self.setup_endpoint(ml_client, endpoint_name)

        log_message("Validating Deployment info...")
        self.setup_deployment(ml_client, deployment_name)

        # showing ml workspace info at console
        show_ml_info(ml_client, self._endpoint_url, self._deployment_name)

    def setup_endpoint(self, ml_client: MLClient, endpoint_name: str):
        try:
            endpoint = ml_client.online_endpoints.get(endpoint_name)
            keys = ml_client.online_endpoints.get_keys(endpoint_name)
        except Exception as e:
            log_message(
                f"Failed to retrieve endpoint information: {e}", level="error"
            )
            raise

        self._endpoint_url = endpoint.scoring_uri
        self._endpoint_key = (
            keys.primary_key
            if endpoint.auth_mode == "key"
            else keys.access_token
        )

        parsed_url = urlparse(self._endpoint_url)
        self.protocol, self.host, self.path = (
            parsed_url.scheme,
            parsed_url.netloc,
            parsed_url.path,
        )

        self._endpoint_name = endpoint_name

    def setup_deployment(self, ml_client: MLClient, deployment_name: str):
        """Function to validating ML Endpoint's Deployment Name

        Args:
            ml_client (MLClient): _description_
            deployment_name (str): _description_
        """
        self._deployment_name = deployment_name

        try:
            _ = ml_client.online_deployments.get(
                name=self._deployment_name, endpoint_name=self._endpoint_name
            )
        except Exception as e:
            log_message(
                f"Failed to retrieve model deployment information: {e}",
                level="error",
            )
            raise

    def exec_api(
        self, msg: str, chat_history_for_ml: List[Dict[str, str]]
    ) -> Union[Dict[str, str], None]:
        """Executes the API call.

        Args:
            msg (str): User question message
            chat_history_for_ml (List[Dict[str, str | Dict]]): ChatHistory Json for MLAPI. The Format is `{ "inputs": {"question": msg}, "outputs": bot_message }`

        Returns:
            tuple ( Union[Dict[str, str], None], int, str ): ("None" | "Result JSON for MLAPI". The Format is `{ "answer", "<HERE RESPONSE MESSAGE>" }`, status_code, status_reason_msg)
        """
        # ML Endpoint Payload example
        payload: Dict[str, Union[str, List[Dict[str, str]]]] = {
            "question": msg,
            "chat_history": chat_history_for_ml,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._endpoint_key}",
            "azureml-model-deployment": self._deployment_name,
            "Accept": "application/json",
        }

        response = requests.post(
            self._endpoint_url,
            json=payload,
            headers=headers,
        )

        try:
            response.raise_for_status()
            log_message(
                f"Got response: {response.status_code} {response.reason}"
            )
            content_type = response.headers.get("Content-Type")
            log_message(f"Response Content-Type: {content_type}")

            try:
                return (
                    response.json(),
                    response.status_code,
                    response.reason,
                )
            except requests.exceptions.JSONDecodeError as e:
                log_message(
                    f"Failed to parse JSON response: {e}", level="error"
                )
                return (None, None, None)

        except requests.HTTPError as e:
            log_message(f"error: {e}", level="error")
            return None

    @abstractmethod
    def respond_simple(
        self,
        msg: str,
        chat_history: List[Dict[str, str]],
        chat_history_for_ml: List[Dict[str, str]],
    ) -> AISimpleResponse:
        """Processes Non-Streaming chat messages in Gradio chatbot. This method must be implemented by subclasses.

        Args:
            msg (str): User input question.

            chat_history (List[Dict[str, str]]): List of ChatHistory Json for AOAI.
                The Format is `[ {"role": "user", "content": "<USER MESSAGE1>"}, {"role": "assistant", "content": "<RESPONSE MESSAGE1>"},
                    {"role": "user", "content": "<USER MESSAGE2>"}, {"role": "assistant", "content": "<RESPONSE MESSAGE2>"} ]`

            chat_history_for_ml (List[Dict[str, str]]): LIST of ChatHistory Json for MLAPI.
                The Format is `[ {"inputs": {"question": "<USER MESSAGE1>"}, "outputs": "<RESPONSE MESSAGE1>"},
                    {"inputs": {"question": "<USER MESSAGE2>"}, "outputs": "<RESPONSE MESSAGE2>"},・・・]`

        Returns:
            (AISimpleResponse): The updated message, chat history, and ML chat history.
        """

        ...

    @abstractmethod
    def respond_stream(
        self,
        msg: str,
        chat_history: List[Dict[str, str]],
        chat_history_for_ml: List[Dict[str, str]],
        delay: float = 0.01,
    ) -> Generator[AISimpleResponse, None, None]:
        """Processes Non-Streaming chat messages in Gradio chatbot. This method must be implemented by subclasses.

        Args:
            msg (str): User input question.

            chat_history (List[Dict[str, str]]): List of ChatHistory Json for AOAI.
                The Format is `[ {"role": "user", "content": "<USER MESSAGE1>"}, {"role": "assistant", "content": "<RESPONSE MESSAGE1>"},
                    {"role": "user", "content": "<USER MESSAGE2>"}, {"role": "assistant", "content": "<RESPONSE MESSAGE2>"} ]`

            chat_history_for_ml (List[Dict[str, str]]): LIST of ChatHistory Json for MLAPI.
                The Format is `[ {"inputs": {"question": "<USER MESSAGE1>"}, "outputs": "<RESPONSE MESSAGE1>"},
                    {"inputs": {"question": "<USER MESSAGE2>"}, "outputs": "<RESPONSE MESSAGE2>"},・・・]`

        Returns:
            (Generator[AISimpleResponse, None]): The updated message, chat history, and ML chat history.
        """

        ...
