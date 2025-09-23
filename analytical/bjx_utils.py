import os
import sys

# Default path to the PARENT directory of the local 'blackjax' package folder.
DEFAULT_BLACKJAX_PARENT_DIR = '../blackjax/'


def setup_blackjax_path(blackjax_parent_dir: str = DEFAULT_BLACKJAX_PARENT_DIR):
    """Add local blackjax repo parent dir to sys.path and import blackjax.

    Returns the imported blackjax module, or exits on failure.
    """
    absolute_path = os.path.abspath(blackjax_parent_dir)
    if os.path.isdir(os.path.join(absolute_path, 'blackjax')):
        if absolute_path not in sys.path:
            sys.path.insert(0, absolute_path)
            print(f"Added {absolute_path} to sys.path")
        else:
            print(f"{absolute_path} already in sys.path")

        try:
            import blackjax  # type: ignore
            print(f"Imported blackjax from: {blackjax.__file__}")
            if hasattr(blackjax, '__version__'):
                print(f"Blackjax version: {blackjax.__version__}")
            return blackjax
        except Exception as exc:  # noqa: BLE001
            print(f"Could not import blackjax from {absolute_path}: {exc}")
            sys.exit(1)
    else:
        print(f"Blackjax directory not found in {absolute_path}. Provide the correct parent directory.")
        sys.exit(1)
