"""Shared fake OpenAI-compatible client for LLM-call tests."""


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)  # each: a response string or an Exception
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome

        class Msg:
            content = outcome

        class Choice:
            message = Msg()

        class Resp:
            choices = [Choice()]

        return Resp()


class FakeClient:
    def __init__(self, outcomes):
        self.completions = FakeCompletions(outcomes)

        class Chat:
            pass

        self.chat = Chat()
        self.chat.completions = self.completions
