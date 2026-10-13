"""A thin client for HotCRP's REST API, for the few calls this repo makes.

Everything else here works from HotCRP's downloads. Two things only the API
can give: a comment's **visibility and author** (the action log records
neither, and the text export has no dates), and a way to **post** into HotCRP
without a browser. `scripts/discussion_nudges.py` needs both.

**Authentication.** A per-user bearer token from HotCRP's Account settings ->
Developer page, read from `HOTCRP_TOKEN`; the site from `HOTCRP_SITE`
(`https://hpca2027.hotcrp.com`). Both live in `.env`, which the Makefile
includes and exports. The token is never printed, not even in an error. Scopes
needed: `comment:read comment:write review:read paper:read submeta:read
other:read`. The API acts **as the token's owner**, so papers that person is
conflicted with are simply absent from every response -- the same hole the
leads download has -- and callers have to count what they did not see.

**Shape.** `get`/`post` return the parsed JSON body. HotCRP answers every
request with an `ok` field; `ok: false` becomes `HotcrpError` carrying the
server's `message_list`, as does any non-2xx status. A 429/5xx is retried the
way DBLP fetches are (`dblp.get_with_retry`, with a short floor -- HotCRP is
not DBLP). Paths are the documented ones, `api/<path>`; `{p}` forms are given
as `comment` with `p=` rather than spelled into the path.

**Snapshots.** With `snapshot_dir` set, every GET's body is written under it,
keyed by path and parameters, and `offline=True` answers from those files
instead of the network. A generate run can be repeated byte-for-byte without
re-reading the site, and the tests never touch the network.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time

import requests

from reviewer_match import dblp

SITE_ENV = "HOTCRP_SITE"
TOKEN_ENV = "HOTCRP_TOKEN"

_USER_AGENT = "HPCA2027-reviewer-match/0.1 (discussion nudges; contact the PC chairs)"

# Message statuses HotCRP uses in `message_list`: 2+ is an error, 1 a warning.
ERROR_STATUS = 2
WARNING_STATUS = 1
# POST/DELETE retries on 429/5xx: 3s doubling, capped, honouring Retry-After.
POST_RETRIES = 7
POST_MAX_BACKOFF = 120


class HotcrpError(Exception):
    """A request HotCRP refused, with its message list attached."""

    def __init__(self, what: str, messages: list[dict] | None = None, status: int | None = None):
        self.messages = messages or []
        self.status = status
        detail = "; ".join(format_message(m) for m in self.messages)
        super().__init__(f"{what}" + (f": {detail}" if detail else ""))


def format_message(m: dict) -> str:
    """One `message_list` entry as text, stripping HotCRP's inline HTML."""
    text = m.get("message") or m.get("text") or ""
    text = re.sub(r"<[^>]+>", "", str(text))
    field = m.get("field")
    return f"{field}: {text}" if field else text


def errors(messages: list[dict]) -> list[dict]:
    return [m for m in messages or () if int(m.get("status", 0)) >= ERROR_STATUS]


def warnings(messages: list[dict]) -> list[dict]:
    return [m for m in messages or () if int(m.get("status", 0)) == WARNING_STATUS]


class Site:
    """One HotCRP site plus the token used to talk to it."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        session: requests.Session | None = None,
        snapshot_dir: str | None = None,
        offline: bool = False,
    ) -> None:
        if not base_url or not token:
            raise ValueError("HotCRP site and token are both required")
        self.base_url = base_url.rstrip("/")
        self._token = token
        self.session = session or requests.Session()
        self.snapshot_dir = snapshot_dir
        self.offline = offline
        if offline and not snapshot_dir:
            raise ValueError("offline mode needs a snapshot directory to answer from")

    @classmethod
    def from_env(cls, env=os.environ, **kwargs) -> "Site":
        site = (env.get(SITE_ENV) or "").strip()
        token = (env.get(TOKEN_ENV) or "").strip()
        missing = [name for name, value in ((SITE_ENV, site), (TOKEN_ENV, token)) if not value]
        if missing:
            raise ValueError(
                f"{' and '.join(missing)} not set: put HOTCRP_SITE=https://hpca2027.hotcrp.com and "
                f"HOTCRP_TOKEN=hct_... (Account settings -> Developer) in .env"
            )
        return cls(site, token, **kwargs)

    # -- transport -----------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self.base_url}/api/{path.lstrip('/')}"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"bearer {self._token}", "User-Agent": _USER_AGENT,
                "Accept": "application/json"}

    def _snapshot_path(self, path: str, params: dict) -> str:
        key = json.dumps([path, sorted((k, str(v)) for k, v in params.items())], sort_keys=True)
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        slug = re.sub(r"[^A-Za-z0-9]+", "_", path).strip("_")
        return os.path.join(self.snapshot_dir, f"{slug}-{digest}.json")

    @staticmethod
    def _clean(params: dict) -> dict:
        out = {}
        for k, v in params.items():
            if v is None:
                continue
            if isinstance(v, bool):
                v = 1 if v else 0
            out[k] = v
        return out

    @staticmethod
    def _body(resp: requests.Response, what: str) -> dict:
        try:
            body = resp.json()
        except ValueError:
            raise HotcrpError(f"{what}: HTTP {resp.status_code}, non-JSON reply", status=resp.status_code)
        if not isinstance(body, dict):
            raise HotcrpError(f"{what}: unexpected reply shape", status=resp.status_code)
        if resp.status_code >= 400 or not body.get("ok", False):
            raise HotcrpError(f"{what}: HTTP {resp.status_code}", body.get("message_list"), resp.status_code)
        return body

    def get(self, path: str, **params) -> dict:
        params = self._clean(params)
        snap = self._snapshot_path(path, params) if self.snapshot_dir else None
        if self.offline:
            if not snap or not os.path.exists(snap):
                raise FileNotFoundError(f"offline: no snapshot for GET {path} {params} (expected {snap})")
            with open(snap, encoding="utf-8") as f:
                return json.load(f)
        resp = self._get_raw(path, params)
        body = self._body(resp, f"GET {path}")
        if snap:
            os.makedirs(self.snapshot_dir, exist_ok=True)
            tmp = snap + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(body, f, indent=1, sort_keys=True)
            os.replace(tmp, snap)
        return body

    def _get_raw(self, path: str, params: dict) -> requests.Response:
        """GET with DBLP-style backoff, but never `raise_for_status`: HotCRP
        answers a refused request with a JSON body worth reading."""
        try:
            return dblp.get_with_retry(
                self.session, self._url(path), params=params, headers=self._headers(),
                backoff_floor=3, max_retries=4, timeout=60,
            )
        except requests.HTTPError as exc:
            if exc.response is not None:
                return exc.response
            raise

    def post(self, path: str, data: dict, *, method: str = "POST") -> dict:
        """POST (form body) or DELETE (query string -- HotCRP does not read a
        DELETE's body, and answered "Comment not found" when the id was there)."""
        if self.offline:
            raise RuntimeError(f"offline: refusing to {method} {path}")
        data = self._clean(data)
        where = {"params": data} if method == "DELETE" else {"data": data}
        for attempt in range(POST_RETRIES + 1):
            resp = self.session.request(method, self._url(path), headers=self._headers(), timeout=60, **where)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < POST_RETRIES:
                # A 509-comment dry run outlasted a 21s backoff: honour
                # Retry-After, and back off long enough to clear the window.
                wait = min(POST_MAX_BACKOFF, 3 * (2 ** attempt))
                try:
                    wait = max(wait, min(POST_MAX_BACKOFF, int(resp.headers.get("Retry-After", 0))))
                except ValueError:
                    pass
                print(f"    HotCRP answered {resp.status_code} to {method} {path}; retrying in {wait}s",
                      file=sys.stderr)
                time.sleep(wait)
                continue
            break
        return self._body(resp, f"{method} {path}")

    # -- the handful of calls this repo makes --------------------------------

    def whoami(self) -> dict:
        """The token's account (`GET /account?email=me`)."""
        return self.get("account", email="me")

    def search(self, q: str, t: str = "s") -> list[int]:
        """Submission ids matching a HotCRP search, in search order."""
        body = self.get("search", q=q, t=t)
        return [int(pid) for pid in body.get("ids") or []]

    def comments(self, q: str, t: str = "s", content: bool = False) -> list[dict]:
        """Every comment the token's owner can see on the papers `q` matches."""
        body = self.get("comments", q=q, t=t, content=content)
        return list(body.get("comments") or [])

    def reviews(self, q: str, t: str = "s") -> list[dict]:
        """Every review (metadata and form) on the papers `q` matches."""
        body = self.get("reviews", q=q, t=t)
        return list(body.get("reviews") or [])

    def post_comment(
        self, pid: int, text: str, *, visibility: str = "rev", topic: str = "rev",
        tags: str = "", notify: bool = True, dry_run: bool = False,
    ) -> dict:
        """Create one comment on paper `pid` (`POST /{p}/comment`, `c=new`).

        `visibility`: admin | pc | rev | au; `topic`: paper | rev | dec. With
        `dry_run` HotCRP validates and reports but saves nothing. `notify`
        False suppresses the mention and follower emails (honoured only for a
        submission the token's owner administers, which a chair does).
        """
        data = {"p": pid, "c": "new", "text": text, "visibility": visibility, "topic": topic,
                "tags": tags or None, "notify": notify, "dry_run": dry_run}
        return self.post(f"{int(pid)}/comment", data)

    def delete_comment(self, pid: int, cid: int, *, notify: bool = False) -> dict:
        return self.post(f"{int(pid)}/comment", {"c": int(cid), "notify": notify}, method="DELETE")


def pid_query(pids, chunk: int = 250):
    """Yield HotCRP search strings naming at most `chunk` papers each, sorted.

    A search of 600 bare numbers is a 3 KB URL; chunking keeps every request
    comfortably short and, because the chunks are sorted, keeps the snapshot
    keys stable from run to run.
    """
    ordered = sorted({int(p) for p in pids})
    for i in range(0, len(ordered), chunk):
        yield " ".join(str(p) for p in ordered[i:i + chunk])
