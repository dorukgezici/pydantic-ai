from __future__ import annotations as _annotations

from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, TypedDict, Union

from ..messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from ..settings import ModelSettings
from ..tools import ToolDefinition
from ..usage import Usage
from . import (
    AgentModel,
    Model,
    StreamedResponse,
    check_allow_model_requests,
)

try:
    import mlx.nn as nn
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from mlx_lm.utils import generate, load
except ImportError as import_error:
    raise ImportError(
        'Please install `mlx_lm` to use MLX-LM models, '
        "you can use the `mlx-ml` optional group — `pip install 'pydantic-ai-slim[mlx-ml]'`"
    ) from import_error


CommonMLXModelNames = Literal[
    'mistralai/Mistral-7B-v0.1',
    'meta-llama/Llama-2-7b-hf',
    'deepseek-ai/deepseek-coder-6.7b-instruct',
    'microsoft/phi-2',
]
"""
For a full list see [github.com/ml-explore/mlx-examples/blob/main/llms/README.md#supported-models).
"""

MLXModelName = Union[CommonMLXModelNames, str]
"""
Since MLX-LM supports lots of models, we explicitly list the most common models but allow any name in the type hints.
"""


class Message(TypedDict):
    """OpenAI-compatible message format."""

    content: str
    role: str


@dataclass(init=False)
class MLXModel(Model):
    """A model that implements MLX-ML for local inference.

    Internally, this uses the [MLX-LM package](https://github.com/ml-explore/mlx-lm) to run models locally on Apple Silicon devices.
    """

    model_name: MLXModelName
    model: nn.Module
    tokenizer: TokenizerWrapper

    def __init__(
        self,
        model_name: MLXModelName,
    ):
        """Initialize an MLX model.

        Args:
            model_name: The name of the MLX model to use. List of models available at
                github.com/ml-explore/mlx-examples/blob/main/llms/README.md#supported-models
        """
        self.model_name = model_name
        self.model, self.tokenizer = load(model_name)

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        """Create an agent model for function calling.

        Currently, MLX models don't support function calling natively, so we format the tools
        as part of the prompt in a format the model can understand.
        """
        check_allow_model_requests()
        return MLXAgentModel(
            model_name=self.model_name,
            model=self.model,
            tokenizer=self.tokenizer,
            allow_text_result=allow_text_result,
            function_tools=function_tools,
            result_tools=result_tools,
        )

    def name(self) -> str:
        return f'mlx-lm:{self.model_name}'


@dataclass
class MLXAgentModel(AgentModel):
    """Implementation of `AgentModel` for MLX-LM models."""

    model_name: str
    model: nn.Module
    tokenizer: TokenizerWrapper

    allow_text_result: bool
    function_tools: list[ToolDefinition]
    result_tools: list[ToolDefinition]

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
    ) -> AsyncIterator[StreamedResponse]:
        """Make a streaming request to the model from Pydantic AI call."""
        # TODO: Implement streamed requests
        raise NotImplementedError('Streamed requests not supported by this MLXAgentModel')
        yield

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
    ) -> tuple[ModelResponse, Usage]:
        """Make a non-streaming request to the MLX-LM model."""
        response = await self._completions_create(messages, False, model_settings)

        return (
            ModelResponse(parts=[TextPart(content=response)], timestamp=datetime.now(timezone.utc)),
            Usage(),
        )

    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[False],
        model_settings: ModelSettings | None,
    ) -> str:
        prompt = self.tokenizer.apply_chat_template(
            conversation=[self._map_message(message) for message in messages],
            add_generation_prompt=True,
        )  # type: ignore
        return generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,  # type: ignore
            # max_tokens=model_settings.get('max_tokens', 1000) if model_settings else 1000,
            # temp=model_settings.get('temperature', 0.7) if model_settings else 0.7,
        )

    @classmethod
    def _map_message(cls, message: ModelMessage) -> Iterable[Message]:
        """Just maps a `pydantic_ai.ModelMessage` to a `Message` TypedDict."""
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    yield Message(content=part.content, role='user')
                elif isinstance(part, SystemPromptPart):
                    yield Message(content=part.content, role='system')

        elif isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, TextPart):
                    yield Message(content=part.content, role='assistant')
                elif isinstance(part, ToolCallPart):
                    # TODO: Implement tool calls
                    pass
