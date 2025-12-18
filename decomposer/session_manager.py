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
from threading import RLock

from neuro_san.client.agent_session_factory import AgentSession
from neuro_san.client.agent_session_factory import AgentSessionFactory


# pylint: disable=too-few-public-methods
class SessionManager:
    """
    Singleton class to manage agent sessions.
    """

    # Global, shared across threads
    _factory_lock: RLock = RLock()
    _factory: AgentSessionFactory | None = None
    _sessions: dict[str, AgentSession] = {}

    @staticmethod
    def get_session(agent_name: str) -> AgentSession:
        """
        Return a shared, thread-safe session for the named agent.
        """
        with SessionManager._factory_lock:
            if SessionManager._factory is None:
                SessionManager._factory = AgentSessionFactory()
            sess = SessionManager._sessions.get(agent_name)
            if sess is None:
                sess = SessionManager._factory.create_session(
                    "direct",
                    agent_name,
                    use_direct=True,
                    metadata={"user_id": os.environ.get("USER")}
                )
                SessionManager._sessions[agent_name] = sess
            return sess
