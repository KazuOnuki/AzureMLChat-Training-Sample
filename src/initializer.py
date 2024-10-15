from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from dotenv import find_dotenv, load_dotenv

from src.utils import get_env_variable, log_message


def initialize_client(filename=".env") -> MLClient:
    """Function to initialize MLClient / Endpoint_Name / Deployment_Name from .env file

    Returns:
        (Tuple): ml_client, endpoint_name, deployment_name
    """
    log_message("Loading .env info...")
    if not load_dotenv(find_dotenv(filename)):
        log_message(
            "Error: .env file not found or couldn't be loaded.", level="error"
        )

    sub_id = get_env_variable("SUBSCRIPTION_ID")
    rg_name = get_env_variable("RESOURCE_GROUP_NAME")
    ws_name = get_env_variable("WORKSPACE_NAME")
    endpoint_name = get_env_variable("ENDPOINT_NAME")
    deployment_name = get_env_variable("DEPLOYMENT_NAME")

    log_message("Getting MLWorkspace info...")

    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=sub_id,
        resource_group_name=rg_name,
        workspace_name=ws_name,
    )

    return ml_client, endpoint_name, deployment_name
