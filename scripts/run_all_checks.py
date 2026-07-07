#!/usr/bin/env python3
"""Master validation script — runs all pre-release checks.

Executes:
1. pytest test suite
2. submission.jsonl schema validation
3. Cache directory verification
4. Deployment configuration checks
5. Overall PASS/FAIL report

Usage:
    python scripts/run_all_checks.py
"""

import json
import subprocess
import sys
from pathlib import Path

# Force UTF-8 stdout on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── ANSI Colors ──────────────────────────────────────────────
class C:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def header(title: str) -> None:
    print(f"\n{C.BOLD}{C.CYAN}{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}{C.RESET}")


def passed(msg: str) -> bool:
    print(f"  {C.GREEN}✓{C.RESET} {msg}")
    return True


def failed(msg: str) -> bool:
    print(f"  {C.RED}✗{C.RESET} {msg}")
    return False


# ── Check 1: Pytest ──────────────────────────────────────────

def check_pytest() -> bool:
    header("Check 1: Test Suite (pytest)")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-q"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        # Print last few lines of output (summary)
        lines = result.stdout.strip().splitlines()
        for line in lines[-10:]:
            print(f"    {line}")

        if result.returncode == 0:
            return passed("All tests passed")
        else:
            # Print stderr if any
            if result.stderr:
                for line in result.stderr.strip().splitlines()[-5:]:
                    print(f"    {C.RED}{line}{C.RESET}")
            return failed(f"Tests failed (exit code {result.returncode})")
    except subprocess.TimeoutExpired:
        return failed("Tests timed out after 120s")
    except FileNotFoundError:
        return failed("pytest not found — install with: pip install pytest")


# ── Check 2: submission.jsonl ────────────────────────────────

def check_submission() -> bool:
    header("Check 2: submission.jsonl Validation")
    submission_path = PROJECT_ROOT / "submission.jsonl"

    if not submission_path.exists():
        return failed("submission.jsonl not found")

    lines = submission_path.read_text(encoding="utf-8").strip().splitlines()
    passed_msg = passed(f"Found {len(lines)} entries")

    required_fields = {"test_id", "body", "cta", "send_as", "suppression_key", "rationale"}
    valid_ctas = {"binary_yes_stop", "open_ended", "none"}
    valid_send_as = {"vera", "merchant_on_behalf"}
    errors = []

    for i, line in enumerate(lines, 1):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"Line {i}: invalid JSON — {e}")
            continue

        missing = required_fields - set(entry.keys())
        if missing:
            errors.append(f"Line {i} ({entry.get('test_id', '?')}): missing {missing}")

        if entry.get("cta") not in valid_ctas:
            errors.append(f"Line {i} ({entry.get('test_id', '?')}): invalid cta '{entry.get('cta')}'")

        if entry.get("send_as") not in valid_send_as:
            errors.append(f"Line {i} ({entry.get('test_id', '?')}): invalid send_as '{entry.get('send_as')}'")

        body = entry.get("body", "")
        if not body or len(body.strip()) < 10:
            errors.append(f"Line {i} ({entry.get('test_id', '?')}): body too short")

    if errors:
        for e in errors[:5]:  # Show first 5 errors
            print(f"    {C.RED}{e}{C.RESET}")
        if len(errors) > 5:
            print(f"    {C.RED}... and {len(errors) - 5} more errors{C.RESET}")
        return failed(f"{len(errors)} validation error(s)")
    else:
        return passed("All entries have valid schema")


# ── Check 3: Cache Directory ─────────────────────────────────

def check_cache() -> bool:
    header("Check 3: Cache Directory")
    cache_dir = PROJECT_ROOT / "cache" / "responses"

    if not cache_dir.exists():
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            passed("Cache directory created")
        except OSError as e:
            return failed(f"Cannot create cache directory: {e}")

    # Check writability
    test_file = cache_dir / "_test_write.tmp"
    try:
        test_file.write_text("test", encoding="utf-8")
        test_file.unlink()
        passed("Cache directory is writable")
    except OSError as e:
        return failed(f"Cache not writable: {e}")

    # Count entries
    count = len(list(cache_dir.glob("*.json")))
    passed(f"Cached responses: {count}")
    return True


# ── Check 4: Deployment Config ───────────────────────────────

def check_deployment() -> bool:
    header("Check 4: Deployment Configuration")
    all_ok = True

    # Dockerfile
    dockerfile = PROJECT_ROOT / "Dockerfile"
    if dockerfile.exists():
        content = dockerfile.read_text(encoding="utf-8")
        if "uvicorn" in content:
            passed("Dockerfile exists and references uvicorn")
        else:
            all_ok = False
            failed("Dockerfile missing uvicorn command")
    else:
        all_ok = False
        failed("Dockerfile not found")

    # docker-compose.yml
    compose = PROJECT_ROOT / "docker-compose.yml"
    if compose.exists():
        passed("docker-compose.yml exists")
    else:
        print(f"  {C.YELLOW}○{C.RESET} docker-compose.yml not found (optional)")

    # render.yaml
    render = PROJECT_ROOT / "render.yaml"
    if render.exists():
        passed("render.yaml exists")
    else:
        print(f"  {C.YELLOW}○{C.RESET} render.yaml not found (optional)")

    # requirements.txt
    req = PROJECT_ROOT / "requirements.txt"
    if req.exists():
        passed("requirements.txt exists")
    else:
        all_ok = False
        failed("requirements.txt not found")

    # .env.example
    env_ex = PROJECT_ROOT / ".env.example"
    if env_ex.exists():
        passed(".env.example exists")
    else:
        all_ok = False
        failed(".env.example not found")

    # LICENSE
    lic = PROJECT_ROOT / "LICENSE"
    if lic.exists():
        passed("LICENSE exists")
    else:
        print(f"  {C.YELLOW}○{C.RESET} LICENSE not found (optional)")

    return all_ok


# ── Check 5: Key Files ───────────────────────────────────────

def check_key_files() -> bool:
    header("Check 5: Key Source Files")
    all_ok = True

    key_files = [
        "app/main.py",
        "app/api/routes.py",
        "app/core/config.py",
        "app/core/exceptions.py",
        "app/llm/groq_client.py",
        "app/services/composer.py",
        "app/services/prompt_builder.py",
        "app/services/output_validator.py",
        "app/services/reply_handler.py",
        "app/services/tick_handler.py",
        "cache/response_cache.py",
        "cache/rate_limiter.py",
        "evaluation/batch_runner.py",
        "evaluation/dataset_loader.py",
        "evaluation/evaluator.py",
        "evaluation/metrics.py",
        "scripts/generate_submission.py",
        "scripts/verify_release.py",
    ]

    missing = []
    for f in key_files:
        path = PROJECT_ROOT / f
        if not path.exists():
            missing.append(f)

    if missing:
        for f in missing:
            failed(f"Missing: {f}")
        all_ok = False
    else:
        passed(f"All {len(key_files)} key files present")

    return all_ok


# ── Main ─────────────────────────────────────────────────────

def main():
    print(f"\n{C.BOLD}{'═' * 50}")
    print(f"  Vera AI Agent — Pre-Release Validation")
    print(f"{'═' * 50}{C.RESET}")
    print(f"  Project: {PROJECT_ROOT}")

    results = {
        "Test Suite": check_pytest(),
        "Submission File": check_submission(),
        "Cache": check_cache(),
        "Deployment": check_deployment(),
        "Key Files": check_key_files(),
    }

    # Summary
    total = len(results)
    pass_count = sum(1 for v in results.values() if v)
    fail_count = total - pass_count

    print(f"\n{C.BOLD}{'═' * 50}")
    print(f"  RESULTS: {pass_count}/{total} checks passed")

    for name, ok in results.items():
        icon = f"{C.GREEN}✓{C.RESET}" if ok else f"{C.RED}✗{C.RESET}"
        print(f"    {icon} {name}")

    if fail_count == 0:
        print(f"\n  {C.GREEN}{C.BOLD}✓ ALL CHECKS PASSED — Ready for release!{C.RESET}")
        print(f"{'═' * 50}\n")
        sys.exit(0)
    else:
        print(f"\n  {C.RED}{C.BOLD}✗ {fail_count} check(s) failed{C.RESET}")
        print(f"{'═' * 50}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
