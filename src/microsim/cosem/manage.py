"""CLI scripts for managing COSEM data."""

import argparse

from microsim.cosem import CosemDataset, clear_cache


def download_dataset(args: argparse.Namespace) -> None:
    dset = CosemDataset.fetch(args.name)
    for key in args.image_keys:
        img = dset.image(name=key)
        img.download(dest=getattr(args, "dest", None))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage COSEM data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Fetch datasets
    fetch_datasets = subparsers.add_parser("fetch_datasets", help="Fetch datasets.")
    fetch_datasets.set_defaults(func=fetch_datasets)

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
    download.set_defaults(func=download_dataset)

    # clear cache
    cache = subparsers.add_parser("clear_cache", help="Clear the cache.")
    cache.set_defaults(func=lambda _: clear_cache())

    # Show image
    show = subparsers.add_parser("show", help="Show an image.")
    show.add_argument("name", help="The name of the image.")
    show.set_defaults(func=show)

    return parser.parse_args()


def main() -> None:
    """Main entry point for the COSEM CLI."""
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
