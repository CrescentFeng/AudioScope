from __future__ import annotations

import argparse
import secrets
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a local, gitignored AudioScope .env")
    parser.add_argument("--output", type=Path, default=Path(".env"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing {args.output}")
    value = "\n".join(
        [
            f"AUDIOSCOPE_DB_PASSWORD={secrets.token_urlsafe(36)}",
            "AUDIOSCOPE_DB_PORT=55432",
            "AUDIOSCOPE_API_PORT=8000",
            "AUDIOSCOPE_DEMO_PORT=7862",
            "",
        ]
    )
    args.output.write_text(value, encoding="utf-8")
    args.output.chmod(0o600)
    print(f"Created {args.output} with mode 0600; the secret was not printed.")


if __name__ == "__main__":
    main()
