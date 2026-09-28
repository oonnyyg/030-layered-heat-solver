import argparse

from .server import serve


def main():
    parser = argparse.ArgumentParser(
        prog="layered_heat",
        description="Layered transient heat solver (JSONL over stdin/stdout).",
    )
    parser.add_argument("--state", required=True, help="state directory (SQLite)")
    args = parser.parse_args()
    serve(args.state)


if __name__ == "__main__":
    main()
