"""Volcengine Ark client for the offline proposal stage (Python standard library)."""
import argparse
from dataclasses import dataclass
import json
import os
import sys
from urllib import error, parse, request

DEFAULT_BASE = "https://ark.cn-beijing.volces.com/api/coding/v3"
DEFAULT_MODEL = "deepseek-v4-pro"


class ArkError(RuntimeError):
    """An API or configuration error safe to display to the caller."""


@dataclass(frozen=True)
class ArkConfig:
    base_url: str
    model: str

    @classmethod
    def from_env(cls):
        base = os.environ.get("ARK_BASE_URL", DEFAULT_BASE).strip().rstrip("/")
        model = os.environ.get("ARK_MODEL", DEFAULT_MODEL).strip()
        parts = parse.urlsplit(base)
        if (parts.scheme != "https" or not parts.hostname or parts.username is not None
                or parts.password is not None or parts.query or parts.fragment):
            raise ArkError("ARK_BASE_URL must be an HTTPS base URL without credentials, query or fragment.")
        if not model:
            raise ArkError("Set ARK_MODEL to a model name or endpoint ID.")
        return cls(base, model)

    @property
    def endpoint(self):
        return self.base_url + "/chat/completions"


class _NoRedirect(request.HTTPRedirectHandler):
    # An authenticated POST belongs to the configured endpoint.
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def chat(prompt, *, max_tokens=1800):
    cfg = ArkConfig.from_env()
    key = os.environ.get("ARK_API_KEY", "").strip()
    if not key:
        raise ArkError("Set ARK_API_KEY in the environment before making a live request.")
    payload = {"model": cfg.model, "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": max_tokens, "thinking": {"type": "disabled"}}
    post = request.Request(cfg.endpoint, data=json.dumps(payload).encode(),
                           headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    try:
        with request.build_opener(_NoRedirect()).open(post, timeout=120) as response:
            obj = json.load(response)
    except error.HTTPError as exc:
        raise ArkError(f"Ark returned HTTP {exc.code}; check credentials, model access and endpoint.") from None
    except (error.URLError, TimeoutError, OSError):
        raise ArkError("Could not reach the configured Ark endpoint within the request timeout.") from None
    except ValueError:
        raise ArkError("Ark returned an invalid JSON response.") from None
    try:
        content = obj["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError
    except (KeyError, IndexError, TypeError, ValueError):
        raise ArkError("Ark returned no assistant text.") from None
    return content, {"model": obj.get("model"), "usage": obj.get("usage"), "response_id": obj.get("id")}


def request_proposal(prompt):
    text, meta = chat(prompt)
    try:
        proposal = json.loads(text[text.index("{"):text.rindex("}") + 1])
        if not isinstance(proposal, dict) or not isinstance(proposal.get("edits"), list):
            raise ValueError
    except ValueError:
        raise ArkError("The planner response must contain a JSON object with an edits array.") from None
    return proposal, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Check local settings only (no request)")
    mode.add_argument("--ping", action="store_true", help="Send one short live chat request")
    args = parser.parse_args()
    try:
        cfg = ArkConfig.from_env()
        if args.check:
            present = bool(os.environ.get("ARK_API_KEY", "").strip())
            print(json.dumps({"endpoint": cfg.endpoint, "model": cfg.model, "key_configured": present}))
            return 0 if present else 1
        _, meta = chat("Reply with OK.", max_tokens=32)
        print(json.dumps({"connected": True, **meta}))
        return 0
    except ArkError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
