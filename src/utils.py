import json
import os
import time
import traceback

from azure.ai.ml import MLClient
from rich import print
from rich.tree import Tree


def log_message(message: str, level: str = "info") -> None:
    """Logs messages with specified severity level.

    Args:
        message (str): The message to log.
        level (str): The severity level of the message ('info' or 'error').
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if level == "error":
        print(f"[red]{timestamp} - ERROR:[/red] {message}")
        print(f"[red]{traceback.format_exc()}[/red]")  # Print stack trace
    else:
        print(f"[cyan]{timestamp} - INFO:[/cyan] {message}")


def get_env_variable(var_name: str) -> str:
    """Retrieves environment variable or raises an error if not found.

    Args:
        var_name (str): The name of the environment variable.

    Returns:
        str: The value of the environment variable.

    Raises:
        ValueError: If the environment variable is not found.
    """
    value = os.getenv(var_name)
    if not value:
        log_message(
            f"Environment variable {var_name} is missing.", level="error"
        )
        raise ValueError(
            f"Error: Required environment variable '{var_name}' is not set."
        )
    return value


def show_ml_info(
    ml_client: MLClient,
    endpoint_url: str,
    deployment_name: str,
):
    """Function to show ML workspace Info

    Args:
        ml_client (MLClient): The Azure Machine Learning client.
        endpoint_name (str): Name of the online endpoint.
        deployment_name (str): Name of the deployment in the endpoint.
    """
    tree = Tree(f"[cyan]Subscription[/cyan]: {ml_client.subscription_id}")

    tree.add(
        f"[cyan]Resource Group[/cyan]: {ml_client.resource_group_name}"
    ).add(
        f"[cyan]Machine Learning Workspace[/cyan]: {ml_client.workspace_name}"
    ).add(
        f"[cyan]Managed Online Endpoint[/cyan]: {endpoint_url}"
    ).add(
        f"[cyan]Deployment[/cyan]: {deployment_name}"
    )

    log_message("**Workspace Info**")
    print(tree)


def format_http_log(call_history: str):
    return f"""
<head>
    <!-- Prism.js -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-core.min.js"></script>
    <!-- Prism.js additional Plugin -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/autoloader/prism-autoloader.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/normalize-whitespace/prism-normalize-whitespace.min.js"></script>
</head>

<body>
    {call_history}
</body>
        """


def create_http_log(
    call_count: int,
    _cls,
    jinput: dict,
    joutput: dict,
    res_status_code: int,
    res_status_reason: str,
):
    return f"""
<span style="color: orange;">#{call_count} API Request</span>
<pre class="language-http" tabindex="0">
<span class="token request-line"><span class="token method property">POST </span><span class="token request-target url">{_cls.path} </span><span class="token http-version property">{_cls.protocol}</span></span>
<span class="token header"><span class="token header-name keyword">Host</span><span class="token punctuation">: </span><span class="token header-value">{_cls.host}</span></span>
<span class="token header"><span class="token header-name keyword">Accept</span><span class="token punctuation">: </span><span class="token header-value">application/json</span></span>
<span class="token header"><span class="token header-name keyword">Authorization</span><span class="token punctuation">: </span><span class="token header-value">Bearer &lt; MASKED_APIKey &gt;</span></span>
<span class="token header"><span class="token header-name keyword">azureml-model-deployment</span><span class="token punctuation">: </span><span class="token header-value">{_cls._deployment_name}</span></span>
<span class="token header"><span class="token header-name keyword">Content-Type</span><span class="token punctuation">: </span><span class="token header-value">application/json</span></span>

{json.dumps(
    jinput,
    indent=4,
    ensure_ascii=False,
)}
</pre>

<span style="color: orange;">#{call_count} Response</span>
<pre class="language-http" tabindex="0">
<span class="token response-status"><span class="token http-version property">HTTP/1.1 </span><span class="token status-code number">{res_status_code} </span><span class="token reason-phrase string">{res_status_reason}</span></span>
<span class="token header"><span class="token header-name keyword">Content-Type</span><span class="token punctuation">: </span><span class="token header-value">application/json</span></span>

{json.dumps(
    joutput,
    indent=4,
    ensure_ascii=False,
)}
</pre>
<hr>
    """
