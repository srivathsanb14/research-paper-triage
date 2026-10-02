"""Optional account backend for Paper Triage: login, per-user state and labels in SQLite.

    python -m server            # http://localhost:8000, also serves the built site (_site/)

The static site works without it (browser-only). When this server is reachable the
web app offers sign-in and syncs each user's profiles, ratings and labels here.
"""
