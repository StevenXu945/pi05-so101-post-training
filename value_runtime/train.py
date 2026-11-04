"""Run RECAP train from the project root or this directory."""
from recap_value.train import main, parse_args, cli, train_step, evaluation_step

__all__ = ['main', 'parse_args', 'cli', 'train_step', 'evaluation_step']

if __name__ == "__main__":
    cli()
