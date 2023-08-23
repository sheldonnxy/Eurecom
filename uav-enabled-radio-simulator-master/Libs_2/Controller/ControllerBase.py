from abc import ABC, abstractmethod
from Libs.Environments.DataCollection import EnvParamStr


class ControllerBase(ABC):
    def __init__(self, env_param=EnvParamStr()):
        self.env_param = env_param
        self.agent_state = None
        self.agent_info = None

