from __future__ import annotations as _annotations

from datetime import timezone

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    UserPromptPart,
)
from pydantic_ai.result import Usage

from ..conftest import IsNow, try_import

with try_import() as imports_successful:
    import mlx.nn as nn
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    from pydantic_ai.models.mlx_lm import MLXModel

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='mlx_lm not installed'),
    pytest.mark.anyio,
]


def test_init():
    m = MLXModel('microsoft/phi-2')
    assert isinstance(m.model, nn.Module)
    assert isinstance(m.tokenizer, TokenizerWrapper)
    assert m.name() == 'mlx-lm:microsoft/phi-2'


async def test_request_simple_success(allow_model_requests: None):
    m = MLXModel('microsoft/phi-2')
    agent = Agent(m)

    result = await agent.run('hello')
    assert isinstance(result.data, str)
    assert len(result.data) > 0
    assert result.usage() == snapshot(Usage(requests=1))

    result = await agent.run('hello', message_history=result.new_messages())
    assert isinstance(result.data, str)
    assert len(result.data) > 0
    assert result.usage() == snapshot(Usage(requests=1))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse.from_text(content=result.data, timestamp=IsNow(tz=timezone.utc)),
            ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse.from_text(content=result.data, timestamp=IsNow(tz=timezone.utc)),
        ]
    )
