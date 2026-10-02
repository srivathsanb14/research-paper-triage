import argparse

import uvicorn

parser = argparse.ArgumentParser(description="Paper Triage server (site + accounts)")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()
uvicorn.run("server.app:app", host=args.host, port=args.port)
