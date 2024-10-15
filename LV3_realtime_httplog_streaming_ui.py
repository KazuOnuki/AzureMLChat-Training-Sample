import time
from pathlib import Path
from typing import Dict, Generator, List

import gradio as gr
from azure.ai.ml import MLClient
from rich import print

from src.chat import AICustomResponse, AISimpleResponse, BaseChatApp
from src.initializer import initialize_client
from src.utils import create_http_log, format_http_log, log_message


class ChatApp(BaseChatApp):
    def __init__(
        self,
        ml_client: MLClient,
        endpoint_name: str,
        deployment_name: str,
    ):
        super().__init__(ml_client, endpoint_name, deployment_name)

    def respond_simple(self) -> AISimpleResponse:
        pass

    def respond_stream(
        self,
        msg: str,
        chat_history: List[Dict[str, str | Dict]],
        chat_history_for_ml: List[Dict[str, str]],
        call_history: str,
        call_log_md_display: str,
        call_count: int,
        delay: float = 0.01,
    ) -> Generator[AICustomResponse, None, None]:
        """Processes Non-Streaming chat messages in Gradio chatbot. This method must be implemented by subclasses.

        Args:
            msg (str): User input question.
            chat_history (List[Dict[str, str]]): List of ChatHistory Json for AOAI.
                The Format is `[ {"role": "user", "content": "<USER MESSAGE1>"}, {"role": "assistant", "content": "<RESPONSE MESSAGE1>"},
                    {"role": "user", "content": "<USER MESSAGE2>"}, {"role": "assistant", "content": "<RESPONSE MESSAGE2>"} ]`

            chat_history_for_ml (List[Dict[str, str | Dict]]): LIST of ChatHistory Json for MLAPI.
                The Format is `[ {"inputs": {"question": "<USER MESSAGE1>"}, "outputs": "<RESPONSE MESSAGE1>"},
                    {"inputs": {"question": "<USER MESSAGE2>"}, "outputs": "<RESPONSE MESSAGE2>"},・・・]`

            delay (float): Processing Interval of output message

            call_history (gr.State): Past call log.
            call_log_md_display (gr.Markdown): Call log markdown.
            call_count (gr.State): API call count.

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

            yield AISimpleResponse(
                bot_message="",
                chat_history=chat_history,
                chat_history_for_ml=chat_history_for_ml,
            )

            return

        res_json, res_status_code, res_status_reason = result

        payload = {"question": msg, "chat_history": chat_history_for_ml}
        call_count += 1

        call_history += create_http_log(
            call_count=call_count,
            _cls=self,
            jinput=payload,
            joutput=res_json,
            res_status_code=res_status_code,
            res_status_reason=res_status_reason,
        )

        call_log_md_display = format_http_log(call_history=call_history)

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

        # NOTE: intetionally run `for` Loop to behave like streaming Chat
        response: Dict[str, str] = {"role": "assistant", "content": ""}
        for message in bot_message:
            time.sleep(delay)
            response["content"] += message or ""
            yield AICustomResponse(
                bot_message="",
                chat_history=chat_history[:-1] + [response],
                chat_history_for_ml=chat_history_for_ml,
                call_history=call_history,
                call_log_md_display=call_log_md_display,
                call_count=call_count,
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

    with (Path(__file__).parent / "assets" / "main.css").open(
        encoding="utf-8"
    ) as fi:
        _css: str = fi.read()

    # theme = "freddyaboulton/dracula_revamped"
    with gr.Blocks(theme="NoCrypt/miku", css=_css) as demo:

        call_count = gr.State(0)
        call_history = gr.State(str())
        chat_history_for_ml = gr.State(list())

        gr.Markdown("# 🐈 Chat with Azure Machine Learning API Endpoint")

        with gr.Row():

            with gr.Column(scale=3, elem_id="chat-area"):
                chat_history = gr.Chatbot(type="messages")
                msg = gr.Textbox()
                clear_chat_btn = gr.Button("Clear Chat", variant="secondary")

            with gr.Column(scale=1, elem_id="chat-info-panel"):
                with gr.Accordion("Information Panel", open=True):
                    with gr.Accordion("API Call Log", open=True):
                        call_log_md_display = gr.HTML(value="")

        clear_chat_btn.click(
            fn=lambda: (list(), str(), str(), list(), str(), 0),
            outputs=[
                chat_history,
                msg,
                call_history,
                chat_history_for_ml,
                call_log_md_display,
                call_count,
            ],
        )

        # Handler for gradio button trigger function
        def handle_response(
            msg,
            chat_history,
            chat_history_for_ml,
            call_history,
            call_log_md_display,
            call_count,
        ):
            response_generator = chat_app.respond_stream(
                msg,
                chat_history,
                chat_history_for_ml,
                call_history,
                call_log_md_display,
                call_count,
            )

            # process each response from generator
            for response in response_generator:
                yield response.bot_message, response.chat_history, response.chat_history_for_ml, response.call_history, response.call_log_md_display, response.call_count,

        msg.submit(
            fn=handle_response,
            inputs=[
                msg,
                chat_history,
                chat_history_for_ml,
                call_history,
                call_log_md_display,
                call_count,
            ],
            outputs=[
                msg,
                chat_history,
                chat_history_for_ml,
                call_history,
                call_log_md_display,
                call_count,
            ],
        )

    demo.launch(debug=True)
