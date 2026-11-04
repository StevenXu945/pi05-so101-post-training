"""Run RECAP label advantage from the project root or this directory."""
from recap_value.label_advantage import main, parse_args, cli

__all__ = ['main', 'parse_args', 'cli']

if __name__ == "__main__":
    cli()
