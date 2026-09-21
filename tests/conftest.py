import pathlib
import sys

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_real_provider_keys(monkeypatch):
    """Les tests ne doivent jamais appeler un fournisseur reel, meme si les cles
    sont exportees dans l'environnement du developpeur."""
    for name in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "TYPESAFE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
