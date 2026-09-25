"""Select the optional paper path without loading model or metric dependencies."""


def dispatch_if_requested(run_paper_method):
    """Consume only an explicit --method; otherwise preserve legacy argv exactly.

    The nine original drivers call this before their existing imports. Importing
    this module requires no PyTorch, vLLM, Transformers, or metric packages.
    """
    import sys

    args = sys.argv[1:]
    option_args = args[:args.index("--")] if "--" in args else args
    if not any(arg == "--method" or arg.startswith("--method=")
               for arg in option_args):
        return

    import argparse

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--method", choices=("legacy", "paper"),
                        default="legacy")
    method_args, remaining = parser.parse_known_args(args)
    sys.argv[1:] = remaining
    if method_args.method == "paper":
        raise SystemExit(run_paper_method(remaining))
