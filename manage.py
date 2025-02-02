#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import json



def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'personal_RAG.settings')
    os.environ.setdefault('DJANGO_CONFIGURATION', 'DEV')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    try:
        with open(".secrets.json", "r") as f:
            secrets = json.load(f)
            for key, value in secrets.items():
                os.environ[key] = value
    except FileNotFoundError:
        print("No .secrets.json found")
    main()
