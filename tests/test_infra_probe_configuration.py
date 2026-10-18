"""Check version probing with mocked Docker/HTTP, without starting services."""
from memlab import envcheck


def test_version_probe_uses_container_postgres_configuration(monkeypatch):
    commands = []

    def run(command, **_):
        commands.append(command)
        if "redis-cli" in command:
            return True, "redis_version:8.0.0\n"
        if "postgres" in command and "sh" in command:
            return True, " 18.3 \n"
        if "ps" in command:
            return True, "synthetic-container"
        if "image" in command:
            return True, "synthetic-image@sha256:example"
        return True, "synthetic-image"

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"version": "1.0.0"}

    class Client:
        def __init__(self, **_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def get(self, _):
            return Response()

    monkeypatch.setattr(envcheck, "_run", run)
    monkeypatch.setattr(envcheck.httpx, "Client", Client)
    versions, _, errors = envcheck._probe_infra_versions_and_digests()
    assert errors == []
    assert versions["postgres"] == "18.3"
    postgres_command = next(cmd for cmd in commands if "postgres" in cmd and "sh" in cmd)
    shell_command = postgres_command[-1]
    assert '"$POSTGRES_USER"' in shell_command
    assert '"$POSTGRES_DB"' in shell_command
    assert "POSTGRES_PASSWORD" not in shell_command
    assert "PG_URI" not in shell_command
