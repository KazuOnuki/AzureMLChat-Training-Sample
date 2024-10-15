# 🐈 Azure ML Chat Application - Training Materials

This project implements a chat interface that interacts with Azure Machine Learning API endpoints using Gradio. It provides three levels of implementation, each with increasing functionality and complexity. This repository is intended **exclusively for external AI training materials**, spanning 1-2 days.

## 🌟 Features

- 🤖 Chat interface using Gradio
- 🔗 Integration with Azure Machine Learning endpoints
- 📡 Three levels of implementation:
  - Level 1: Non-streaming chat
  - Level 2: Streaming chat
  - Level 3: Streaming chat with real-time HTTP logs
- 🎨 Customizable UI with CSS styling
- 🔐 Secure handling of API keys and endpoints

## 🚀 Getting Started

### Prerequisites

Before getting started, ensure you have the following:

- An active Azure Subscription
- An Azure Machine Learning Workspace resource
- A deployment of [Promptflow](https://github.com/microsoft/promptflow/tree/main/examples/flows/chat/chat-with-wikipedia) to an **Azure Managed Online Endpoint** using [Azure ML Studio UI](http://ml.azure.com)
- Python >= 3.10 installed locally
- Azure CLI for managing resources

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/KazuOnuki/azureml-chatapi-example.git
   cd azure-ml-chat-training-materials
   ```

2. Install the required packages and Jump in Venv:
   ```bash
   python -m venv .venv
   pip install -r requirements.txt
   .\.venv\Scripts\Activate
   ```

3. Set up your `.env` file with the following variables:
   ```bash
   SUBSCRIPTION_ID=your_subscription_id
   RESOURCE_GROUP_NAME=your_resource_group
   WORKSPACE_NAME=your_workspace_name
   ENDPOINT_NAME=your_endpoint_name
   DEPLOYMENT_NAME=your_deployment_name
   ```

    > Make sure the `ENDPOINT_NAME` and `DEPLOYMENT_NAME` correspond to the Promptflow deployment mentioned in the prerequisites.

### 🖥️ Usage

Run one of the following scripts based on your desired level of functionality:

#### Level 1: Non-streaming Chat

```bash
python LV1_nonstreaming_ui.py
```

#### Level 2: Streaming Chat

```bash
python LV2_streaming_ui.py
```

#### Level 3: Streaming Chat with Real-time HTTP Logs

```bash
python LV3_realtime_httplog_streaming_ui.py
```

After running the script, open the provided URL in your web browser to interact with the chat interface.

## 🏗️ Project Structure

- `src/`
  - `chat.py`: Base chat application class and response models
  - `initializer.py`: MLClient initialization
  - `utils.py`: Utility functions for logging and HTTP formatting
- `LV1_nonstreaming_ui.py`: Non-streaming chat implementation
- `LV2_streaming_ui.py`: Streaming chat implementation
- `LV3_realtime_httplog_streaming_ui.py`: Streaming chat with real-time HTTP logs
- `assets/main.css`: Custom CSS for the Gradio interface

## 🛠️ Customization

You can customize the appearance of the chat interface by modifying the `assets/main.css` file. The project uses the "NoCrypt/miku" theme for Gradio, but you can change this in the `LV3_realtime_httplog_streaming_ui.py` file.


## 🔐 API Key Security

To securely handle API keys, ensure the `.env` file is properly configured. Additionally, avoid sharing this file or committing it to public repositories to prevent exposure of sensitive information.

## 📖 Lessons from the Code

This project offers several valuable lessons on best practices in Python and AI-driven applications:

1. **Abstract Base Classes (ABC) and Interface Definition**
   The code uses Python's `abc` module to define an abstract base class (`BaseChatApp`). This ensures that any subclass must implement the `respond_simple` and `respond_stream` methods.
   **Takeaway**: Abstract base classes enforce an interface while allowing flexibility in subclass implementation, promoting consistency across different implementations.

2. **Structured Data Models with Pydantic**
   Response models (`AISimpleResponse`, `AICustomResponse`) are defined using Pydantic, which provides data validation and parsing.
   **Takeaway**: Pydantic ensures that incoming and outgoing data matches the expected structure, improving reliability and debuggability.

3. **Separation of Concerns and Modularity**
   The code separates concerns by modularizing tasks like API communication (`exec_api`), endpoint setup (`setup_endpoint`), and deployment validation (`setup_deployment`).
   **Takeaway**: Clear code structure promotes maintainability and clarity. Each core functionality is separated into distinct methods, making the code easy to follow.

4. **Exception Handling and Logging**
   Robust error handling is demonstrated with external resources like Azure Machine Learning endpoints. The code logs meaningful error messages via the `log_message` function.
   **Takeaway**: Exception handling with proper logging ensures that issues can be traced without crashing the application.

5. **Azure Machine Learning Integration**
   The code interacts with Azure Machine Learning services, retrieving endpoint information and performing API calls. It uses `MLClient` to manage endpoints and deployments.
   **Takeaway**: The code offers a practical example of integrating cloud services into an AI application.

6. **API Call with Payload and Headers**
   The `exec_api` function structures an API request with a JSON payload and custom headers, including authentication. It validates both request and response.
   **Takeaway**: This demonstrates how to interact with RESTful APIs, handle requests, and parse JSON responses.

7. **Streaming and Non-Streaming Response Handling**
   The abstract methods `respond_simple` and `respond_stream` outline how the application handles non-blocking (streaming) and blocking interactions.
   **Takeaway**: Handling both synchronous and asynchronous responses is vital for modern interactive applications.

8. **Real-World Example of State Management**
   Chat history and contextual data are stored and managed across multiple API calls, demonstrating effective state management.
   **Takeaway**: Managing state is essential for building intelligent, contextual chatbots.

9. **Usage of Python’s typing Module for Strong Typing**
   The `typing` module provides type hints, making the code easier to read and maintain.
   **Takeaway**: Strong typing catches errors early and enhances collaboration on large projects.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
