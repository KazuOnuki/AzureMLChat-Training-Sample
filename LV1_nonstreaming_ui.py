from typing import Dict, Generator, List

import gradio as gr
from azure.ai.ml import MLClient

from src.chat import AISimpleResponse, BaseChatApp
from src.initializer import initialize_client
from src.utils import log_message


class ChatApp(BaseChatApp):
    def __init__(
        self,
        ml_client: MLClient,
        endpoint_name: str,
        deployment_name: str,
    ):
        super().__init__(ml_client, endpoint_name, deployment_name)

    def respond_stream(self) -> Generator[AISimpleResponse, None, None]:
        pass

    def respond_simple(
        self,
        msg: str,
        chat_history: List[Dict[str, str | Dict]],
        chat_history_for_ml: List[Dict[str, str]],
    ) -> AISimpleResponse:
        """Processes Non-Streaming chat messages in Gradio chatbot. This method must be implemented by subclasses.

        Args:
            msg (str): User input question.
            chat_history (List[Dict[str, str]]): List of ChatHistory Json for AOAI.
                The Format is `[ {"role": "user", "content": "<USER MESSAGE1>"}, {"role": "assistant", "content": "<RESPONSE MESSAGE1>"},
                    {"role": "user", "content": "<USER MESSAGE2>"}, {"role": "assistant", "content": "<RESPONSE MESSAGE2>"} ]`

            chat_history_for_ml (List[Dict[str, str | Dict]]): LIST of ChatHistory Json for MLAPI.
                The Format is `[ {"inputs": {"question": "<USER MESSAGE1>"}, "outputs": "<RESPONSE MESSAGE1>"},
                    {"inputs": {"question": "<USER MESSAGE2>"}, "outputs": "<RESPONSE MESSAGE2>"},・・・]`

        Returns:
            (AISimpleResponse): The updated message, chat history, and ML chat history.
        """
        log_message(f"Calling ML OnlineEndpoint...")

        if not (
            result := self.exec_api(
                msg=msg, chat_history_for_ml=chat_history_for_ml
            )
        ):
            log_message(
                "No valid response received from the API.", level="error"
            )
            return

        res_json, _, _ = result
        bot_message: str = res_json.get("answer", "<EMPTY>")

        # NOTE: List of ChatHistory Json for *Azure OpenAI*
        chat_history.append({"role": "user", "content": msg})
        chat_history.append({"role": "assistant", "content": bot_message})

        # NOTE: LIST of ChatHistory Json for *MLAPI*.
        chat_history_for_ml.append(
            {
                "inputs": {"question": msg},
                "outputs": bot_message,
            }
        )

        return AISimpleResponse(
            # NOTE: we have to set empty str for Gradio.Textbox after receiving response
            bot_message="",
            chat_history=chat_history,
            chat_history_for_ml=chat_history_for_ml,
        )


if __name__ == "__main__":
    ml_client, endpoint_name, deployment_name = initialize_client(
        filename=".env"
    )

    # NOTE: call your defined ChatApp(BaseChatApp) class
    chat_app = ChatApp(
        ml_client=ml_client,
        endpoint_name=endpoint_name,
        deployment_name=deployment_name,
    )

    # NOTE: launch ui
    with gr.Blocks() as demo:
        chat_history_for_ml = gr.State(list())
        chat_history = gr.Chatbot(type="messages")
        msg = gr.Textbox()

        clear = gr.ClearButton([msg, chat_history])

        # Handler for gradio button trigger function
        def handle_response(msg, chat_history, chat_history_for_ml):
            response = chat_app.respond_simple(
                msg, chat_history, chat_history_for_ml
            )
            return (
                response.bot_message,
                response.chat_history,
                response.chat_history_for_ml,
            )

        msg.submit(
            fn=handle_response,
            inputs=[msg, chat_history, chat_history_for_ml],
            outputs=[msg, chat_history, chat_history_for_ml],
        )

    demo.launch()
