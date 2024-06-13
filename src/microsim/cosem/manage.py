"""CLI scripts for managing COSEM data.

python -m microsim.cosem.manage --help
"""

import argparse

try:
    from rich import print
except ImportError:
    pass

from microsim.cosem import CosemDataset, bucket_cache, clear_cache


def _clear_cache(args: argparse.Namespace) -> None:
    if not bucket_cache().exists():
        print("Cache is empty.")
        return

    numfiles = len(list(bucket_cache().rglob("*")))
    clear_cache()
    print(f"Deleted {numfiles} files from the cache.")


def _show(args: argparse.Namespace) -> None:
    dset = CosemDataset.fetch(args.name)
    dset.show(image_keys=args.image_keys, level=args.level, bin_mode=args.bin_mode)


def _download_dataset(args: argparse.Namespace) -> None:  # pragma: no cover
    dset = CosemDataset.fetch(args.name)
    for key in args.image_keys:
        try:
            img = dset.image(name=key)
        except ValueError as e:
            print(f"Error: {e}")
            continue
        dest = getattr(args, "dest", None)
        max_level = getattr(args, "max_level", None)
        img.download(dest=dest, max_level=max_level)


def _print_info(args: argparse.Namespace) -> None:
    dset = CosemDataset.fetch(args.name)
    print(dset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage COSEM data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Fetch datasets
    info = subparsers.add_parser("info", help="Fetch datasets.")
    info.add_argument("name", help="The name of the dataset.")
    info.set_defaults(func=_print_info)

    # Fetch views
    fetch_views = subparsers.add_parser("fetch_views", help="Fetch views.")
    fetch_views.set_defaults(func=fetch_views)

    # Fetch organelles
    fetch_organelles = subparsers.add_parser(
        "fetch_organelles", help="Fetch organelles."
    )
    fetch_organelles.set_defaults(func=fetch_organelles)

    # Download image
    download = subparsers.add_parser("download", help="Download an image.")
    download.add_argument("name", help="The name of the image.")
    download.add_argument(
        "image_keys", nargs="+", help="The name of the image to download."
    )
    download.add_argument(
        "--dest", help="The destination directory to download the image to."
    )
    download.add_argument(
        "--max-level",
        type=int,
        default=0,
        help="The maximum level of the image to download.",
    )
    download.set_defaults(func=_download_dataset)

    # clear cache
    cache = subparsers.add_parser("clear_cache", help="Clear the cache.")
    cache.set_defaults(func=_clear_cache)

    # Show image
    show = subparsers.add_parser("show", help="Show an image.")
    show.add_argument("name", help="The name of the image.")
    show.add_argument("image_keys", nargs="+", help="The name of the image to show.")
    show.add_argument(
        "--level", type=int, help="The level of the image to show.", default=1
    )
    show.add_argument(
        "--bin-mode",
        choices=["mode", "sum", "auto"],
        default="auto",
        help="The binning mode to use.",
    )
    show.set_defaults(func=_show)

    return parser.parse_args()


def main() -> None:
    """Main entry point for the COSEM CLI."""
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
