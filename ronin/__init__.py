"""ronin: bring a RunPod vLLM pod up and down on demand and point a coding agent at it."""

__version__ = "0.1.0"

# RunPod's proxy (Cloudflare) returns 403 to urllib's default User-Agent.
USER_AGENT = f"ronin/{__version__}"
