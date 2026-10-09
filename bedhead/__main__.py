"""PyInstaller entrypoint: absolute import so the frozen binary can run cli.main."""
from bedhead.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
