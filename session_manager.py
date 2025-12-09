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

import os
import threading

from neuro_san.client.agent_session_factory import AgentSession
from neuro_san.client.agent_session_factory import AgentSessionFactory

AGENTS_PORT = 30011

# Global, shared across threads
_factory_lock = threading.RLock()
_factory: AgentSessionFactory | None = None
_sessions: dict[str, AgentSession] = {}


def _get_session(agent_name: str) -> AgentSession:
    """Return a shared, thread-safe session for the named agent."""
    global _factory
    with _factory_lock:
        if _factory is None:
            _factory = AgentSessionFactory()
        sess = _sessions.get(agent_name)
        if sess is None:
            sess = _factory.create_session(
                "direct", agent_name, "localhost", AGENTS_PORT, False, {"user_id": os.environ.get("USER")}
            )
            _sessions[agent_name] = sess
        return sess
