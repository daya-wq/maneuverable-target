__all__ = [
    "HideAndSeek",
    "HideAndSeek_envgen",
    "HideAndSeek_deploy",
]


def __getattr__(name):
    if name == "HideAndSeek":
        from .hideandseek import HideAndSeek

        return HideAndSeek
    if name == "HideAndSeek_envgen":
        from .hideandseek_envgen import HideAndSeek_envgen

        return HideAndSeek_envgen
    if name == "HideAndSeek_deploy":
        from .hideandseek_deploy import HideAndSeek_deploy

        return HideAndSeek_deploy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
