# app/logger.py
from rich.console import Console
from rich.traceback import install
from rich import pretty

# Enable pretty tracebacks and pretty formatting
install(show_locals=False)
pretty.install()

# Global console logger for the entire app
console = Console()
