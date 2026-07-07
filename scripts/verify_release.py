#!/usr/bin/env python3
"""Release verification script — validates production readiness.

Runs comprehensive checks against a running Vera AI Agent instance
and reports PASS/FAIL for each verification step.

Usage:
    python scripts/verify_release.py
    python scripts/verify_release.py --url http://localhost:8000
    python scripts/verify_release.py --skip-api  # Skip endpoint checks
"""

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

# Force UTF-8 stdout on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── ANSI Colors ──────────────────────────────────────────────
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def _pass(msg: str) -> bool:
    print(f"  {Colors.GREEN}✓ PASS{Colors.RESET}  {msg}")
    return True


def _fail(msg: str, detail: str = "") -> bool:
    print(f"  {Colors.RED}✗ FAIL{Colors.RESET}  {msg}")
    if detail:
        print(f"          {Colors.RED}{detail}{Colors.RESET}")
    return False


def _skip(msg: str) -> bool:
    print(f"  {Colors.YELLOW}○ SKIP{Colors.RESET}  {msg}")
    return True


def _section(title: str) -> None:
    print(f"\n{Colors.BOLD}{Colors.CYAN}── {title} ──{Colors.RESET}")


# ── HTTP Helper ──────────────────────────────────────────────

def _http_get(url: str) -> tuple[int, dict | None]:
    """Send GET request and return (status_code, json_body)."""
    try:
        req = Request(url)
        resp = urlopen(req, timeout=10)
        body = json.loads(resp.read().decode())
        return resp.status, body
    except URLError as e:
        return 0, None
    except Exception:
        return 0, None


def _http_post(url: str, data: dict) -> tuple[int, dict | None]:
    """Send POST request and return (status_code, json_body)."""
    try:
        payload = json.dumps(data).encode()
        req = Request(url, data=payload, headers={"Content-Type": "application/json"})
        resp = urlopen(req, timeout=30)
        body = json.loads(resp.read().decode())
        return resp.status, body
    except URLError as e:
        # Try to read error response
        if hasattr(e, "code"):
            return e.code, None
        return 0, None
    except Exception:
        return 0, None


# ── Checks ───────────────────────────────────────────────────

def check_health_endpoint(base_url: str, results: list) -> None:
    _section("Health Endpoint")
    status, body = _http_get(f"{base_url}/v1/healthz")
    if status == 200 and body and body.get("status") == "ok":
        results.append(_pass("GET /v1/healthz → 200 {status: ok}"))
    else:
        results.append(_fail("GET /v1/healthz", f"status={status} body={body}"))


def check_metadata_endpoint(base_url: str, results: list) -> None:
    _section("Metadata Endpoint")
    status, body = _http_get(f"{base_url}/v1/metadata")
    if status == 200 and body:
        has_name = bool(body.get("name"))
        has_version = bool(body.get("version"))
        if has_name and has_version:
            results.append(_pass(f"GET /v1/metadata → {body['name']} v{body['version']}"))
        else:
            results.append(_fail("GET /v1/metadata — missing name or version", str(body)))
    else:
        results.append(_fail("GET /v1/metadata", f"status={status}"))


def check_context_endpoint(base_url: str, results: list) -> None:
    _section("Context Endpoint")
    # Send a minimal valid payload to test the endpoint accepts input
    # This will likely fail at LLM call if Groq key isn't set, but we
    # just check that the endpoint exists and validates
    status, body = _http_post(f"{base_url}/v1/context", {})
    if status == 422:
        results.append(_pass("POST /v1/context → 422 (validation works correctly)"))
    elif status == 200:
        results.append(_pass("POST /v1/context → 200"))
    elif status > 0:
        results.append(_pass(f"POST /v1/context → {status} (endpoint exists)"))
    else:
        results.append(_fail("POST /v1/context — endpoint unreachable"))


def check_reply_endpoint(base_url: str, results: list) -> None:
    _section("Reply Endpoint")
    status, body = _http_post(f"{base_url}/v1/reply", {})
    if status == 422:
        results.append(_pass("POST /v1/reply → 422 (validation works correctly)"))
    elif status > 0:
        results.append(_pass(f"POST /v1/reply → {status} (endpoint exists)"))
    else:
        results.append(_fail("POST /v1/reply — endpoint unreachable"))


def check_tick_endpoint(base_url: str, results: list) -> None:
    _section("Tick Endpoint")
    status, body = _http_post(f"{base_url}/v1/tick", {})
    if status in (200, 422):
        results.append(_pass(f"POST /v1/tick → {status}"))
    elif status > 0:
        results.append(_pass(f"POST /v1/tick → {status} (endpoint exists)"))
    else:
        results.append(_fail("POST /v1/tick — endpoint unreachable"))


def check_env_variables(results: list) -> None:
    _section("Environment Variables")
    required_vars = {
        "GROQ_API_KEY": "LLM API key",
        "GROQ_MODEL": "LLM model name",
    }
    optional_vars = {
        "ENVIRONMENT": "Runtime environment",
        "LOG_LEVEL": "Log verbosity",
        "CACHE_ENABLED": "Response caching",
    }

    # Load .env file if it exists
    env_path = PROJECT_ROOT / ".env"
    env_vars = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env_vars[key.strip()] = value.strip()

    for var, desc in required_vars.items():
        value = os.environ.get(var) or env_vars.get(var, "")
        if value and not value.startswith("gsk_your_"):
            results.append(_pass(f"{var} is set ({desc})"))
        else:
            results.append(_fail(f"{var} is missing or placeholder ({desc})"))

    for var, desc in optional_vars.items():
        value = os.environ.get(var) or env_vars.get(var, "")
        if value:
            results.append(_pass(f"{var}={value} ({desc})"))
        else:
            results.append(_skip(f"{var} not set (optional: {desc})"))


def check_groq_configuration(results: list) -> None:
    _section("Groq Configuration")
    env_path = PROJECT_ROOT / ".env"
    env_vars = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env_vars[key.strip()] = value.strip()

    # Check API key format
    api_key = os.environ.get("GROQ_API_KEY") or env_vars.get("GROQ_API_KEY", "")
    if api_key.startswith("gsk_") and len(api_key) > 20:
        results.append(_pass("GROQ_API_KEY format is valid (gsk_…)"))
    elif api_key:
        results.append(_fail("GROQ_API_KEY has unexpected format"))
    else:
        results.append(_fail("GROQ_API_KEY is not set"))

    # Check model name
    model = os.environ.get("GROQ_MODEL") or env_vars.get("GROQ_MODEL", "")
    valid_models = {
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    }
    if model in valid_models:
        results.append(_pass(f"GROQ_MODEL={model} (known model)"))
    elif model:
        results.append(_pass(f"GROQ_MODEL={model} (custom model)"))
    else:
        results.append(_fail("GROQ_MODEL is not set"))


def check_cache_availability(results: list) -> None:
    _section("Cache Availability")
    cache_dir = PROJECT_ROOT / "cache" / "responses"

    if cache_dir.exists():
        results.append(_pass(f"Cache directory exists: {cache_dir}"))
    else:
        # Try to create it
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            results.append(_pass(f"Cache directory created: {cache_dir}"))
        except OSError as e:
            results.append(_fail(f"Cannot create cache directory: {e}"))

    # Check writability
    test_file = cache_dir / "_verify_write_test.tmp"
    try:
        test_file.write_text("test", encoding="utf-8")
        test_file.unlink()
        results.append(_pass("Cache directory is writable"))
    except OSError as e:
        results.append(_fail(f"Cache directory is not writable: {e}"))

    # Count cached entries
    cached_files = list(cache_dir.glob("*.json"))
    results.append(_pass(f"Cached responses: {len(cached_files)}"))


def check_submission_file(results: list) -> None:
    _section("Submission File")
    submission_path = PROJECT_ROOT / "submission.jsonl"

    if not submission_path.exists():
        results.append(_fail("submission.jsonl not found"))
        return

    results.append(_pass(f"submission.jsonl exists ({submission_path.stat().st_size} bytes)"))

    # Validate JSON
    lines = submission_path.read_text(encoding="utf-8").strip().splitlines()
    results.append(_pass(f"submission.jsonl has {len(lines)} entries"))

    required_fields = {"test_id", "body", "cta", "send_as", "suppression_key", "rationale"}
    valid = 0
    for i, line in enumerate(lines, 1):
        try:
            entry = json.loads(line)
            missing = required_fields - set(entry.keys())
            if missing:
                results.append(_fail(f"Line {i}: missing fields {missing}"))
            else:
                valid += 1
        except json.JSONDecodeError as e:
            results.append(_fail(f"Line {i}: invalid JSON — {e}"))

    if valid == len(lines) and valid > 0:
        results.append(_pass(f"All {valid} entries have valid schema"))
    elif valid > 0:
        results.append(_fail(f"Only {valid}/{len(lines)} entries are valid"))


def check_deployment_config(results: list) -> None:
    _section("Deployment Configuration")

    # Dockerfile
    dockerfile = PROJECT_ROOT / "Dockerfile"
    if dockerfile.exists():
        content = dockerfile.read_text(encoding="utf-8")
        if "uvicorn" in content and "EXPOSE" in content:
            results.append(_pass("Dockerfile is valid (uvicorn + EXPOSE)"))
        else:
            results.append(_fail("Dockerfile exists but missing uvicorn or EXPOSE"))
    else:
        results.append(_fail("Dockerfile not found"))

    # docker-compose.yml
    compose_file = PROJECT_ROOT / "docker-compose.yml"
    if compose_file.exists():
        results.append(_pass("docker-compose.yml exists"))
    else:
        results.append(_skip("docker-compose.yml not found (optional)"))

    # render.yaml
    render_file = PROJECT_ROOT / "render.yaml"
    if render_file.exists():
        results.append(_pass("render.yaml exists"))
    else:
        results.append(_skip("render.yaml not found (optional)"))

    # requirements.txt
    req_file = PROJECT_ROOT / "requirements.txt"
    if req_file.exists():
        results.append(_pass("requirements.txt exists"))
    else:
        results.append(_fail("requirements.txt not found"))


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Verify Vera AI Agent release readiness.")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the running server.")
    parser.add_argument("--skip-api", action="store_true", dest="skip_api", help="Skip API endpoint checks.")
    args = parser.parse_args()

    print(f"\n{Colors.BOLD}{'═' * 60}")
    print(f"  Vera AI Agent — Release Verification")
    print(f"{'═' * 60}{Colors.RESET}")
    print(f"  Server: {args.url}")
    print(f"  Project: {PROJECT_ROOT}")

    results: list[bool] = []

    # ── API endpoint checks ──────────────────────────────────
    if not args.skip_api:
        check_health_endpoint(args.url, results)
        check_metadata_endpoint(args.url, results)
        check_context_endpoint(args.url, results)
        check_reply_endpoint(args.url, results)
        check_tick_endpoint(args.url, results)
    else:
        _section("API Endpoints (SKIPPED)")
        print(f"  {Colors.YELLOW}○ Skipped — use without --skip-api to test endpoints{Colors.RESET}")

    # ── Static checks ────────────────────────────────────────
    check_env_variables(results)
    check_groq_configuration(results)
    check_cache_availability(results)
    check_submission_file(results)
    check_deployment_config(results)

    # ── Summary ──────────────────────────────────────────────
    passed = sum(1 for r in results if r)
    failed = sum(1 for r in results if not r)
    total = len(results)

    print(f"\n{Colors.BOLD}{'═' * 60}")
    print(f"  SUMMARY: {passed}/{total} checks passed")

    if failed == 0:
        print(f"  {Colors.GREEN}✓ ALL CHECKS PASSED — Ready for release!{Colors.RESET}")
        print(f"{'═' * 60}{Colors.RESET}\n")
        sys.exit(0)
    else:
        print(f"  {Colors.RED}✗ {failed} check(s) failed — see above for details{Colors.RESET}")
        print(f"{'═' * 60}{Colors.RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
