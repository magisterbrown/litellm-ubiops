from typing import Optional
import json

from litellm.utils import ModelResponse, StreamingChoices
import httpx
import ubiops

from .base import BaseLLM

class UbiOps(BaseLLM):
    _client_session: Optional[httpx.Client] = None
    _aclient_session: Optional[httpx.AsyncClient] = None

    def __init__(self) -> None:
        super().__init__()
        self.idx = 0

    def completion(self, msg: dict, endpoint: str, req_field: str, resp_field: str, api_key: str):
        super().completion()
        project, deployment, version = endpoint.split('/')

        client = ubiops.ApiClient(ubiops.Configuration(api_key={"Authorization": api_key}))
        api = ubiops.CoreApi(client)

        request_i = api.deployment_version_requests_create(
            project_name=project,
            deployment_name=deployment,
            version=version,
            data={req_field: json.dumps(msg)}
        )

        self.idx+=1
        ch = {
            "delta": {"content": request_i.result[resp_field], "role": "assistant"},
            "finish_reason": "stop",
            "index": self.idx,
        }

        return iter([ModelResponse(stream=True, id=str(self.idx), choices=[StreamingChoices(**ch)])])


