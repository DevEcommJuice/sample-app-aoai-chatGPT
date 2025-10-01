import copy
import json
import os
import logging
import uuid
import httpx
import asyncio
from quart import (
    Blueprint,
    Quart,
    jsonify,
    make_response,
    request,
    send_from_directory,
    render_template,
    current_app,
)

from openai import AsyncAzureOpenAI
from azure.identity.aio import (
    DefaultAzureCredential,
    get_bearer_token_provider
)

from backend.auth.auth_utils import get_authenticated_user_details
from backend.security.ms_defender_utils import get_msdefender_user_json
from backend.history.cosmosdbservice import CosmosConversationClient
from backend.settings import (
    app_settings,
    MINIMUM_SUPPORTED_AZURE_OPENAI_PREVIEW_API_VERSION
)
from backend.utils import (
    format_as_ndjson,
    format_stream_response,
    format_non_streaming_response,
    convert_to_pf_format,
    format_pf_non_streaming_response,
)

# ---------------------------------------------------------------------
# App / Blueprints
# ---------------------------------------------------------------------
bp = Blueprint("routes", __name__, static_folder="static", template_folder="static")
cosmos_db_ready = asyncio.Event()


def create_app():
    app = Quart(__name__)
    app.register_blueprint(bp)
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    @app.before_serving
    async def init():
        try:
            app.cosmos_conversation_client = await init_cosmosdb_client()
            cosmos_db_ready.set()
        except Exception:
            logging.exception("Failed to initialize CosmosDB client")
            app.cosmos_conversation_client = None
            raise

    return app


# ---------------------------------------------------------------------
# Static / UI
# ---------------------------------------------------------------------
@bp.route("/")
async def index():
    return await render_template(
        "index.html",
        title=app_settings.ui.title,
        favicon=app_settings.ui.favicon
    )


@bp.route("/favicon.ico")
async def favicon():
    return await bp.send_static_file("favicon.ico")


@bp.route("/assets/<path:path>")
async def assets(path):
    return await send_from_directory("static/assets", path)


# ---------------------------------------------------------------------
# Logging / Debug
# ---------------------------------------------------------------------
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)

USER_AGENT = "GitHubSampleWebApp/AsyncAzureOpenAI/1.0.0"

# ---------------------------------------------------------------------
# Frontend settings (expuestos al FE)
# ---------------------------------------------------------------------
frontend_settings = {
    "auth_enabled": app_settings.base_settings.auth_enabled,
    "feedback_enabled": (
        app_settings.chat_history and app_settings.chat_history.enable_feedback
    ),
    "ui": {
        "title": app_settings.ui.title,
        "logo": app_settings.ui.logo,
        "chat_logo": app_settings.ui.chat_logo or app_settings.ui.logo,
        "chat_title": app_settings.ui.chat_title,
        "chat_description": app_settings.ui.chat_description,
        "show_share_button": app_settings.ui.show_share_button,
        "show_chat_history_button": app_settings.ui.show_chat_history_button,
    },
    "sanitize_answer": app_settings.base_settings.sanitize_answer,
    "oyd_enabled": app_settings.base_settings.datasource_type,
}

# Defender
MS_DEFENDER_ENABLED = os.environ.get("MS_DEFENDER_ENABLED", "true").lower() == "true"

# Herramientas para Function Calling remoto
azure_openai_tools = []
azure_openai_available_tools = []


# ---------------------------------------------------------------------
# Azure OpenAI client
# ---------------------------------------------------------------------
async def init_openai_client():
    try:
        # Versión API mínima (preview)
        if (
            app_settings.azure_openai.preview_api_version
            < MINIMUM_SUPPORTED_AZURE_OPENAI_PREVIEW_API_VERSION
        ):
            raise ValueError(
                f"The minimum supported Azure OpenAI preview API version is "
                f"'{MINIMUM_SUPPORTED_AZURE_OPENAI_PREVIEW_API_VERSION}'"
            )

        # Endpoint
        if (
            not app_settings.azure_openai.endpoint and
            not app_settings.azure_openai.resource
        ):
            raise ValueError("AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_RESOURCE is required")

        endpoint = (
            app_settings.azure_openai.endpoint
            if app_settings.azure_openai.endpoint
            else f"https://{app_settings.azure_openai.resource}.openai.azure.com/"
        )

        # Auth (API Key o Entra ID)
        aoai_api_key = app_settings.azure_openai.key
        ad_token_provider = None
        if not aoai_api_key:
            logging.debug("No AZURE_OPENAI_KEY found, using Azure Entra ID auth")
            # Creamos el token provider con credencial administrada
            async with DefaultAzureCredential() as credential:
                ad_token_provider = get_bearer_token_provider(
                    credential,
                    "https://cognitiveservices.azure.com/.default"
                )

        # Deployment (nombre del deployment, no del modelo)
        deployment = app_settings.azure_openai.model
        if not deployment:
            raise ValueError("AZURE_OPENAI_MODEL is required")

        # Headers
        default_headers = {"x-ms-useragent": USER_AGENT}

        # Descarga de metadatos de tools (si procede)
        if app_settings.azure_openai.function_call_azure_functions_enabled:
            tools_base_url = getattr(
                app_settings.azure_openai,
                "function_call_azure_functions_tools_base_url",
                None
            )
            tools_key = getattr(
                app_settings.azure_openai,
                "function_call_azure_functions_tools_key",
                None
            )
            if tools_base_url and tools_key:
                url = f"{tools_base_url}?code={tools_key}"
                async with httpx.AsyncClient() as client:
                    r = await client.get(url)
                if r.status_code == httpx.codes.OK:
                    azure_openai_tools.extend(json.loads(r.text))
                    for tool in azure_openai_tools:
                        azure_openai_available_tools.append(tool["function"]["name"])
                else:
                    logging.error(
                        "Error getting OpenAI Function Call tools metadata: %s",
                        r.status_code
                    )

        # Cliente
        return AsyncAzureOpenAI(
            api_version=app_settings.azure_openai.preview_api_version,
            api_key=aoai_api_key,
            azure_ad_token_provider=ad_token_provider,
            default_headers=default_headers,
            azure_endpoint=endpoint,
        )
    except Exception:
        logging.exception("Exception in Azure OpenAI initialization")
        raise


# ---------------------------------------------------------------------
# Llamada remota a Azure Functions para Function Calling
# ---------------------------------------------------------------------
async def openai_remote_azure_function_call(function_name, function_args):
    if app_settings.azure_openai.function_call_azure_functions_enabled is not True:
        return

    # Soporta tanto *_tool_* como *_tools_* en settings (robustez)
    base_url = getattr(
        app_settings.azure_openai,
        "function_call_azure_functions_tool_base_url",
        None
    ) or getattr(
        app_settings.azure_openai,
        "function_call_azure_functions_tools_base_url",
        None
    )
    key = getattr(
        app_settings.azure_openai,
        "function_call_azure_functions_tool_key",
        None
    ) or getattr(
        app_settings.azure_openai,
        "function_call_azure_functions_tools_key",
        None
    )

    if not base_url or not key:
        logging.error("Azure Functions tool base URL or key not configured")
        return

    url = f"{base_url}?code={key}"
    headers = {"content-type": "application/json"}
    body = {"tool_name": function_name, "tool_arguments": json.loads(function_args)}

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=body, headers=headers)
    resp.raise_for_status()
    return resp.text


# ---------------------------------------------------------------------
# Cosmos DB (historial de conversación)
# ---------------------------------------------------------------------
async def init_cosmosdb_client():
    cosmos_conversation_client = None
    if app_settings.chat_history:
        try:
            cosmos_endpoint = (
                f"https://{app_settings.chat_history.account}.documents.azure.com:443/"
            )

            if not app_settings.chat_history.account_key:
                async with DefaultAzureCredential() as cred:
                    credential = cred
            else:
                credential = app_settings.chat_history.account_key

            cosmos_conversation_client = CosmosConversationClient(
                cosmosdb_endpoint=cosmos_endpoint,
                credential=credential,
                database_name=app_settings.chat_history.database,
                container_name=app_settings.chat_history.conversations_container,
                enable_message_feedback=app_settings.chat_history.enable_feedback,
            )
        except Exception:
            logging.exception("Exception in CosmosDB initialization")
            cosmos_conversation_client = None
            raise
    else:
        logging.debug("CosmosDB not configured")

    return cosmos_conversation_client


# ---------------------------------------------------------------------
# Preparación de payload para Chat Completions
# ---------------------------------------------------------------------
def prepare_model_args(request_body, request_headers):
    request_messages = request_body.get("messages", [])
    messages = []
    if not app_settings.datasource:
        messages = [
            {
                "role": "system",
                "content": app_settings.azure_openai.system_message
            }
        ]

    for message in request_messages:
        if not message:
            continue
        role = message.get("role")
        if role == "user":
            messages.append({"role": "user", "content": message.get("content")})
        elif role in {"assistant", "function", "tool"}:
            m = {"role": role, "content": message.get("content")}
            if "name" in message:
                m["name"] = message["name"]
            if "function_call" in message:
                m["function_call"] = message["function_call"]
            if "context" in message:
                try:
                    m["context"] = json.loads(message["context"])
                except Exception:
                    # si no es JSON válido, lo ignoramos
                    pass
            messages.append(m)

    user_security_context = None
    if MS_DEFENDER_ENABLED:
        authenticated_user_details = get_authenticated_user_details(request_headers)
        application_name = app_settings.ui.title
        user_security_context = get_msdefender_user_json(
            authenticated_user_details, request_headers, application_name
        )

    model_args = {
        "messages": messages,
        "temperature": app_settings.azure_openai.temperature,
        "max_tokens": app_settings.azure_openai.max_tokens,
        "top_p": app_settings.azure_openai.top_p,
        "stop": app_settings.azure_openai.stop_sequence,
        "stream": app_settings.azure_openai.stream,
        "model": app_settings.azure_openai.model,  # deployment name
    }

    if messages and messages[-1]["role"] == "user":
        if app_settings.azure_openai.function_call_azure_functions_enabled and azure_openai_tools:
            model_args["tools"] = azure_openai_tools

        if app_settings.datasource:
            model_args["extra_body"] = {
                "data_sources": [
                    app_settings.datasource.construct_payload_configuration(
                        request=request
                    )
                ]
            }

    # Copia "limpia" para logs (sin secretos)
    model_args_clean = copy.deepcopy(model_args)
    if model_args_clean.get("extra_body"):
        secret_params = [
            "key",
            "connection_string",
            "embedding_key",
            "encoded_api_key",
            "api_key",
        ]
        params = model_args_clean["extra_body"]["data_sources"][0]["parameters"]
        for s in secret_params:
            if params.get(s):
                params[s] = "*****"

        auth = params.get("authentication", {})
        for f in list(auth.keys()):
            if f in secret_params:
                auth[f] = "*****"

        emb_dep = params.get("embedding_dependency", {})
        if "authentication" in emb_dep:
            for f in list(emb_dep["authentication"].keys()):
                if f in secret_params:
                    emb_dep["authentication"][f] = "*****"

    if model_args.get("extra_body") is None:
        model_args["extra_body"] = {}
    if user_security_context:
        model_args["extra_body"]["user_security_context"] = user_security_context.to_dict()

    logging.debug("REQUEST BODY: %s", json.dumps(model_args_clean, indent=4))
    return model_args


# ---------------------------------------------------------------------
# Promptflow (opcional)
# ---------------------------------------------------------------------
async def promptflow_request(req):
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {app_settings.promptflow.api_key}",
        }
        logging.debug("Setting timeout to %s", app_settings.promptflow.response_timeout)
        async with httpx.AsyncClient(timeout=float(app_settings.promptflow.response_timeout)) as client:
            pf_obj = convert_to_pf_format(
                req,
                app_settings.promptflow.request_field_name,
                app_settings.promptflow.response_field_name
            )
            resp = await client.post(
                app_settings.promptflow.endpoint,
                json={
                    app_settings.promptflow.request_field_name: pf_obj[-1]["inputs"][app_settings.promptflow.request_field_name],
                    "chat_history": pf_obj[:-1],
                },
                headers=headers,
            )
        rj = resp.json()
        rj["id"] = req["messages"][-1]["id"]
        return rj
    except Exception as e:
        logging.error("An error occurred while making promptflow_request: %s", e)


# ---------------------------------------------------------------------
# Function calling (no streaming)
# ---------------------------------------------------------------------
async def process_function_call(response):
    response_message = response.choices[0].message
    messages = []

    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            # Solo ejecutamos funciones declaradas en tools metadata
            if tool_call.function.name not in azure_openai_available_tools:
                continue

            function_response = await openai_remote_azure_function_call(
                tool_call.function.name, tool_call.function.arguments
            )

            messages.append(
                {
                    "role": response_message.role,
                    "function_call": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                    "content": None,
                }
            )
            messages.append(
                {
                    "role": "function",
                    "name": tool_call.function.name,
                    "content": function_response,
                }
            )
        return messages
    return None


# ---------------------------------------------------------------------
# Chat: envío base
# ---------------------------------------------------------------------
async def send_chat_request(request_body, request_headers):
    # Filtra mensajes 'tool' (no válidos en chat/completions)
    request_body["messages"] = [
        m for m in request_body.get("messages", []) if m.get("role") != "tool"
    ]
    model_args = prepare_model_args(request_body, request_headers)

    try:
        azure_openai_client = await init_openai_client()
        raw = await azure_openai_client.chat.completions.with_raw_response.create(**model_args)
        response = raw.parse()
        apim_request_id = raw.headers.get("apim-request-id")
        return response, apim_request_id
    except Exception:
        logging.exception("Exception in send_chat_request")
        raise


# ---------------------------------------------------------------------
# Chat: completo (no streaming)
# ---------------------------------------------------------------------
async def complete_chat_request(request_body, request_headers):
    if app_settings.base_settings.use_promptflow:
        response = await promptflow_request(request_body)
        history_metadata = request_body.get("history_metadata", {})
        return format_pf_non_streaming_response(
            response,
            history_metadata,
            app_settings.promptflow.response_field_name,
            app_settings.promptflow.citations_field_name
        )

    response, apim_request_id = await send_chat_request(request_body, request_headers)
    history_metadata = request_body.get("history_metadata", {})
    non_streaming_response = format_non_streaming_response(response, history_metadata, apim_request_id)

    if app_settings.azure_openai.function_call_azure_functions_enabled:
        function_response = await process_function_call(response)
        if function_response:
            request_body["messages"].extend(function_response)
            response, apim_request_id = await send_chat_request(request_body, request_headers)
            history_metadata = request_body.get("history_metadata", {})
            non_streaming_response = format_non_streaming_response(response, history_metadata, apim_request_id)

    return non_streaming_response


# ---------------------------------------------------------------------
# Function calling (streaming)
# ---------------------------------------------------------------------
class AzureOpenaiFunctionCallStreamState:
    def __init__(self):
        self.tool_calls = []
        self.tool_name = ""
        self.tool_arguments_stream = ""
        self.current_tool_call = None
        self.function_messages = []
        self.streaming_state = "INITIAL"  # INITIAL, STREAMING, COMPLETED


async def process_function_call_stream(completionChunk, state, request_body, request_headers, history_metadata, apim_request_id):
    if not (hasattr(completionChunk, "choices") and completionChunk.choices):
        return state.streaming_state

    delta = completionChunk.choices[0].delta

    # En curso
    if delta.tool_calls and state.streaming_state in ["INITIAL", "STREAMING"]:
        state.streaming_state = "STREAMING"
        for tc in delta.tool_calls:
            if tc.id:
                if state.current_tool_call:
                    state.tool_arguments_stream += tc.function.arguments or ""
                    state.current_tool_call["tool_arguments"] = state.tool_arguments_stream
                    state.tool_arguments_stream = ""
                    state.tool_name = ""
                    state.tool_calls.append(state.current_tool_call)

                state.current_tool_call = {
                    "tool_id": tc.id,
                    "tool_name": tc.function.name if not state.tool_name else state.tool_name
                }
            else:
                state.tool_arguments_stream += tc.function.arguments or ""

    # Fin del stream de tool_calls
    elif delta.tool_calls is None and state.streaming_state == "STREAMING":
        state.current_tool_call["tool_arguments"] = state.tool_arguments_stream
        state.tool_calls.append(state.current_tool_call)

        for tool_call in state.tool_calls:
            tool_response = await openai_remote_azure_function_call(
                tool_call["tool_name"], tool_call["tool_arguments"]
            )
            state.function_messages.append({
                "role": "assistant",
                "function_call": {
                    "name": tool_call["tool_name"],
                    "arguments": tool_call["tool_arguments"]
                },
                "content": None
            })
            state.function_messages.append({
                "tool_call_id": tool_call["tool_id"],
                "role": "function",
                "name": tool_call["tool_name"],
                "content": tool_response,
            })

        state.streaming_state = "COMPLETED"

    return state.streaming_state


async def stream_chat_request(request_body, request_headers):
    response, apim_request_id = await send_chat_request(request_body, request_headers)
    history_metadata = request_body.get("history_metadata", {})

    async def generate(apim_request_id, history_metadata):
        if app_settings.azure_openai.function_call_azure_functions_enabled:
            state = AzureOpenaiFunctionCallStreamState()
            async for chunk in response:
                stream_state = await process_function_call_stream(
                    chunk, state, request_body, request_headers, history_metadata, apim_request_id
                )

                if stream_state == "INITIAL":
                    yield format_stream_response(chunk, history_metadata, apim_request_id)

                if stream_state == "COMPLETED":
                    request_body["messages"].extend(state.function_messages)
                    function_response, apim_request_id = await send_chat_request(request_body, request_headers)
                    async for fchunk in function_response:
                        yield format_stream_response(fchunk, history_metadata, apim_request_id)
        else:
            async for chunk in response:
                yield format_stream_response(chunk, history_metadata, apim_request_id)

    return generate(apim_request_id=apim_request_id, history_metadata=history_metadata)


# ---------------------------------------------------------------------
# Orquestación de conversación (streaming o no)
# ---------------------------------------------------------------------
async def conversation_internal(request_body, request_headers):
    try:
        if app_settings.azure_openai.stream and not app_settings.base_settings.use_promptflow:
            gen = await stream_chat_request(request_body, request_headers)
            resp = await make_response(format_as_ndjson(gen))
            resp.timeout = None
            resp.mimetype = "application/json-lines"
            return resp
        else:
            result = await complete_chat_request(request_body, request_headers)
            return jsonify(result)
    except Exception as ex:
        logging.exception(ex)
        status = getattr(ex, "status_code", 500)
        return jsonify({"error": str(ex)}), status


# ---------------------------------------------------------------------
# Rutas HTTP
# ---------------------------------------------------------------------
@bp.route("/conversation", methods=["POST"])
async def conversation():
    if not request.is_json:
        return jsonify({"error": "request must be json"}), 415
    request_json = await request.get_json()
    return await conversation_internal(request_json, request.headers)


@bp.route("/frontend_settings", methods=["GET"])
def get_frontend_settings():
    try:
        return jsonify(frontend_settings), 200
    except Exception:
        logging.exception("Exception in /frontend_settings")
        return jsonify({"error": "internal error"}), 500


# ---------------------------------------------------------------------
# API de historial (Cosmos)
# ---------------------------------------------------------------------
@bp.route("/history/generate", methods=["POST"])
async def add_conversation():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id")

    try:
        if not current_app.cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        history_metadata = {}
        if not conversation_id:
            title = await generate_title(request_json["messages"])
            conversation_dict = await current_app.cosmos_conversation_client.create_conversation(
                user_id=user_id, title=title
            )
            conversation_id = conversation_dict["id"]
            history_metadata["title"] = title
            history_metadata["date"] = conversation_dict["createdAt"]

        messages = request_json["messages"]
        if not messages or messages[-1]["role"] != "user":
            raise Exception("No user message found")

        createdMessageValue = await current_app.cosmos_conversation_client.create_message(
            uuid=str(uuid.uuid4()),
            conversation_id=conversation_id,
            user_id=user_id,
            input_message=messages[-1],
        )
        if createdMessageValue == "Conversation not found":
            raise Exception(
                "Conversation not found for the given conversation ID: " + conversation_id + "."
            )

        request_body = await request.get_json()
        history_metadata["conversation_id"] = conversation_id
        request_body["history_metadata"] = history_metadata
        return await conversation_internal(request_body, request.headers)

    except Exception:
        logging.exception("Exception in /history/generate")
        return jsonify({"error": "internal error"}), 500


@bp.route("/history/update", methods=["POST"])
async def update_conversation():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id")

    try:
        if not current_app.cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        if not conversation_id:
            raise Exception("No conversation_id found")

        messages = request_json["messages"]
        if not messages or messages[-1].get("role") != "assistant":
            raise Exception("No bot messages found")

        if len(messages) > 1 and messages[-2].get("role") == "tool":
            await current_app.cosmos_conversation_client.create_message(
                uuid=str(uuid.uuid4()),
                conversation_id=conversation_id,
                user_id=user_id,
                input_message=messages[-2],
            )

        await current_app.cosmos_conversation_client.create_message(
            uuid=messages[-1]["id"],
            conversation_id=conversation_id,
            user_id=user_id,
            input_message=messages[-1],
        )

        return jsonify({"success": True}), 200

    except Exception:
        logging.exception("Exception in /history/update")
        return jsonify({"error": "internal error"}), 500


@bp.route("/history/message_feedback", methods=["POST"])
async def update_message():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    request_json = await request.get_json()
    message_id = request_json.get("message_id")
    message_feedback = request_json.get("message_feedback")

    try:
        if not message_id:
            return jsonify({"error": "message_id is required"}), 400
        if not message_feedback:
            return jsonify({"error": "message_feedback is required"}), 400

        updated_message = await current_app.cosmos_conversation_client.update_message_feedback(
            user_id, message_id, message_feedback
        )
        if updated_message:
            return jsonify({
                "message": f"Successfully updated message with feedback {message_feedback}",
                "message_id": message_id,
            }), 200
        else:
            return jsonify({
                "error": f"Unable to update message {message_id}. It either does not exist or the user does not have access to it."
            }), 404

    except Exception:
        logging.exception("Exception in /history/message_feedback")
        return jsonify({"error": "internal error"}), 500


@bp.route("/history/delete", methods=["DELETE"])
async def delete_conversation():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id")

    try:
        if not conversation_id:
            return jsonify({"error": "conversation_id is required"}), 400
        if not current_app.cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        await current_app.cosmos_conversation_client.delete_messages(conversation_id, user_id)
        await current_app.cosmos_conversation_client.delete_conversation(user_id, conversation_id)

        return jsonify({
            "message": "Successfully deleted conversation and messages",
            "conversation_id": conversation_id,
        }), 200

    except Exception:
        logging.exception("Exception in /history/delete")
        return jsonify({"error": "internal error"}), 500


@bp.route("/history/list", methods=["GET"])
async def list_conversations():
    await cosmos_db_ready.wait()
    offset = request.args.get("offset", 0)
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    if not current_app.cosmos_conversation_client:
        raise Exception("CosmosDB is not configured or not working")

    conversations = await current_app.cosmos_conversation_client.get_conversations(
        user_id, offset=offset, limit=25
    )
    if not isinstance(conversations, list):
        return jsonify({"error": f"No conversations for {user_id} were found"}), 404

    return jsonify(conversations), 200


@bp.route("/history/read", methods=["POST"])
async def get_conversation():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "conversation_id is required"}), 400

    if not current_app.cosmos_conversation_client:
        raise Exception("CosmosDB is not configured or not working")

    conversation = await current_app.cosmos_conversation_client.get_conversation(user_id, conversation_id)
    if not conversation:
        return jsonify({
            "error": f"Conversation {conversation_id} was not found. It either does not exist or the logged in user does not have access to it."
        }), 404

    conversation_messages = await current_app.cosmos_conversation_client.get_messages(user_id, conversation_id)
    messages = [
        {
            "id": msg["id"],
            "role": msg["role"],
            "content": msg["content"],
            "createdAt": msg["createdAt"],
            "feedback": msg.get("feedback"),
        }
        for msg in conversation_messages
    ]
    return jsonify({"conversation_id": conversation_id, "messages": messages}), 200


@bp.route("/history/rename", methods=["POST"])
async def rename_conversation():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "conversation_id is required"}), 400

    if not current_app.cosmos_conversation_client:
        raise Exception("CosmosDB is not configured or not working")

    conversation = await current_app.cosmos_conversation_client.get_conversation(user_id, conversation_id)
    if not conversation:
        return jsonify({
            "error": f"Conversation {conversation_id} was not found. It either does not exist or the logged in user does not have access to it."
        }), 404

    title = request_json.get("title")
    if not title:
        return jsonify({"error": "title is required"}), 400

    conversation["title"] = title
    updated = await current_app.cosmos_conversation_client.upsert_conversation(conversation)
    return jsonify(updated), 200


@bp.route("/history/delete_all", methods=["DELETE"])
async def delete_all_conversations():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    try:
        if not current_app.cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        conversations = await current_app.cosmos_conversation_client.get_conversations(
            user_id, offset=0, limit=None
        )
        if not conversations:
            return jsonify({"error": f"No conversations for {user_id} were found"}), 404

        for c in conversations:
            await current_app.cosmos_conversation_client.delete_messages(c["id"], user_id)
            await current_app.cosmos_conversation_client.delete_conversation(user_id, c["id"])

        return jsonify({"message": f"Successfully deleted conversation and messages for user {user_id}"}), 200

    except Exception:
        logging.exception("Exception in /history/delete_all")
        return jsonify({"error": "internal error"}), 500


@bp.route("/history/clear", methods=["POST"])
async def clear_messages():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id")

    try:
        if not conversation_id:
            return jsonify({"error": "conversation_id is required"}), 400
        if not current_app.cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        await current_app.cosmos_conversation_client.delete_messages(conversation_id, user_id)
        return jsonify({
            "message": "Successfully deleted messages in conversation",
            "conversation_id": conversation_id,
        }), 200
    except Exception:
        logging.exception("Exception in /history/clear_messages")
        return jsonify({"error": "internal error"}), 500


@bp.route("/history/ensure", methods=["GET"])
async def ensure_cosmos():
    await cosmos_db_ready.wait()
    if not app_settings.chat_history:
        return jsonify({"error": "CosmosDB is not configured"}), 404

    try:
        success, err = await current_app.cosmos_conversation_client.ensure()
        if not current_app.cosmos_conversation_client or not success:
            return jsonify({"error": err or "CosmosDB is not configured or not working"}), 500
        return jsonify({"message": "CosmosDB is configured and working"}), 200
    except Exception as e:
        logging.exception("Exception in /history/ensure")
        cosmos_exception = str(e)
        if "Invalid credentials" in cosmos_exception:
            return jsonify({"error": cosmos_exception}), 401
        elif "Invalid CosmosDB database name" in cosmos_exception:
            return jsonify({
                "error": f"{cosmos_exception} {app_settings.chat_history.database} for account {app_settings.chat_history.account}"
            }), 422
        elif "Invalid CosmosDB container name" in cosmos_exception:
            return jsonify({
                "error": f"{cosmos_exception}: {app_settings.chat_history.conversations_container}"
            }), 422
        else:
            return jsonify({"error": "CosmosDB is not working"}), 500


# ---------------------------------------------------------------------
# Título automático para historial
# ---------------------------------------------------------------------
async def generate_title(conversation_messages) -> str:
    title_prompt = (
        "Summarize the conversation so far into a 4-word or less title. "
        "Do not use any quotation marks or punctuation. "
        "Do not include any other commentary or description."
    )

    messages = [{"role": msg["role"], "content": msg["content"]} for msg in conversation_messages]
    messages.append({"role": "user", "content": title_prompt})

    try:
        azure_openai_client = await init_openai_client()
        response = await azure_openai_client.chat.completions.create(
            model=app_settings.azure_openai.model,
            messages=messages,
            temperature=1,
            max_tokens=64
        )
        return response.choices[0].message.content
    except Exception:
        logging.exception("Exception while generating title")
        return messages[-2]["content"]


# ---------------------------------------------------------------------
# WSGI/ASGI entrypoint
# ---------------------------------------------------------------------
app = create_app()
