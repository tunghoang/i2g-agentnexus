import threading
import time
from random import uniform
from datetime import datetime

def _is_killed(agent):
    return agent['killed'] == 1

_EXPIRE_TIME = 3 * 60
def _is_expired(agent):
    now = datetime.now()
    tl = agent['tl']
    timedelta = now - tl
    seconds = timedelta.total_seconds()
    return seconds > _EXPIRE_TIME

class Cleaner:
    def __init__(self, app_server):
        self.agents = app_server.agents
        self.stopped = False
    def __main_loop(self):
        while not self.stopped:
            time.sleep(1)
            if uniform(0.0, 1.0) < 0.05:
                print("Cleaner running")
            to_be_killed = []
            for a in self.agents:
                agent = self.agents[a]
                if _is_killed(agent):
                    del agent['agent']
                    to_be_killed.append(a)
                elif _is_expired(agent):
                    agent['killed'] = 1
            for agentid in to_be_killed:
                del self.agents[agentid]

    def run(self):
        self.t = threading.Thread(target=self.__main_loop)
        self.stopped = False
        self.t.start()
    def stop(self):
        self.stopped = True
        self.t.join()

def cleaner_create(app_server):
    return Cleaner(app_server)
