import asyncio
import numpy as np
import pytest

from src.orchestrator import Orchestrator, AgentState

@pytest.mark.asyncio
async def test_orchestrator_initialization(mock_config, mocker):
    mocker.patch("src.orchestrator.AudioCapture")
    mocker.patch("src.orchestrator.AudioPlayback")
    mocker.patch("src.orchestrator.LLMClient")
    
    orchestrator = Orchestrator(mock_config)
    assert orchestrator.state == AgentState.IDLE
    assert not orchestrator._running
