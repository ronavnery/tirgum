"""`tirgum serve` starts the web app; anything else runs the pipeline from the terminal."""

import argparse
import sys


def main() -> None:
    if sys.argv[1:2] == ["serve"]:
        a = argparse.ArgumentParser(prog="tirgum serve", description="Start the Tirgum web app.")
        a.add_argument("--host", default="127.0.0.1",
                       help="address to listen on (0.0.0.0 to allow other machines, e.g. in Docker)")
        a.add_argument("--port", type=int, default=8420)
        a.add_argument("--no-open", action="store_true", help="don't open a browser window")
        args = a.parse_args(sys.argv[2:])
        from .server import serve

        serve(args.host, args.port, open_browser=not args.no_open)
    else:
        from .pipeline import main as run_cli

        run_cli()


if __name__ == "__main__":
    main()
