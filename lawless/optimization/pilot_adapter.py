from time import monotonic
from lawless.world.config import EnvironmentConfig
from lawless.environment import reset, step
from lawless.controllers.memory import MemoryController
from lawless.optimization.evaluator import load_parameters


class PilotTimeout(TimeoutError):
    pass


class PilotAdapter:
    def __init__(self, deadline, max_steps=2400):
        self.deadline, self.max_steps = deadline, max_steps
    def check(self):
        if monotonic() >= self.deadline:
            raise PilotTimeout("120-minute deadline exceeded")

    def run_episode(self, vector, seed):
        self.check()
        config = EnvironmentConfig(seed=int(seed))
        config.validate()
        state, obs, _ = reset(config, num_worlds=1, num_agents=1)
        controller = MemoryController(seed=7)
        load_parameters(controller, vector)
        controller.reset_episode()
        success, steps = False, 0
        for _ in range(self.max_steps):
            self.check()
            actions = controller.act(obs)
            state, obs, reward, terminated, truncated, info = step(state, actions)
            steps += 1
            success |= bool(reward.reshape(-1)[0].item())
            if bool(terminated.reshape(-1)[0].item()) or bool(truncated.reshape(-1)[0].item()):
                break

        return {"seed": int(seed), "success": success, "steps": steps}


    def evaluate(self, vector, seeds, on_episode_start):
        scores, details = [], []
        for seed in seeds:
            self.check()
            on_episode_start()
            result = self.run_episode(vector, seed)
            details.append(result)
            scores.append(float(result["success"]))
        return sum(scores) / len(scores), details

class PilotAccounting:
    def __init__(self):
        self.candidates = 0
        self.episodes = 0
    def candidate_start(self):
        self.candidates += 1
    def episode_start(self):
        self.episodes += 1
