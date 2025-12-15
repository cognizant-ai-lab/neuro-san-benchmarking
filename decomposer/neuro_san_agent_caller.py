# Copyright © 2025 Cognizant Technology Solutions Corp, www.cognizant.com.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# END COPYRIGHT

from typing import Any

import logging
from os import getpid
from threading import get_ident

from neuro_san.client.agent_session_factory import AgentSession
from neuro_san.client.streaming_input_processor import StreamingInputProcessor

from decomposer.agent_caller import AgentCaller
from decomposer.solver_parsing import SolverParsing


class NeuroSanAgentCaller(AgentCaller):
    """
    Generic interface for calling an agent
    """

    def __init__(self, agent_session: AgentSession,
                 parsing: SolverParsing = None,
                 name: str = None):
        """
        Constructor

        :param agent_session: The agent session
        :param parsing: The SolverParsing instance to use (if any) to extract the final answer
        :param name: The name of the agent
        """
        self.agent_session: AgentSession = agent_session
        self.solver_parsing: SolverParsing = parsing
        self.name: str = name

    def get_name(self) -> str:
        """
        Get the name of the agent

        :return: The name of the agent
        """
        if self.name is not None:
            return self.name
        return f"{self.agent_session}"

    def call_agent(self, text: str, timeout_ms: float = 100000.0) -> str:
        """
        Call a single agent with given text, return its response.
        """
        # Set up the chat state for the request
        chat_state: dict[str, Any] = {
            "last_chat_response": None,
            "prompt": "",
            "timeout": timeout_ms,
            "num_input": 0,
            "user_input": text,
            "sly_data": None,
            "chat_filter": {"chat_filter_type": "MAXIMAL"},
        }

        use_name: str = self.get_name()
        logging.debug(f"call_agent({use_name}): sending {len(text)} chars")

        # Call the agent
        inp = StreamingInputProcessor("DEFAULT", self._tmpfile("program_mode_thinking"), self.agent_session, None)
        chat_state = inp.process_once(chat_state)

        # Parse the response
        resp: str = chat_state.get("last_chat_response") or ""
        logging.debug(f"call_agent({use_name}): received {len(resp)} chars")
        if self.solver_parsing is not None:
            resp = self.solver_parsing.extract_final(resp)

        return resp

    # Unique temp file per *call*
    def _tmpfile(self, stem: str) -> str:
        return f"/tmp/{stem}_{getpid()}_{get_ident()}.txt"
