"""Train S3GM on the JHU 125^3 cross-cube protocol.

The 2-D S3GM adapter validates a (num_x, num_y) grid, which a 125^3 cube fails,
so the 3-D adapter from :mod:`methods.s3gm.s3gm3d` is registered for the
``s3gm`` name and the shared generative trainer is run unchanged.
"""
from methods.common.registry import register_adapter
from methods.s3gm.s3gm3d import S3GM3DAdapter

register_adapter("s3gm", S3GM3DAdapter)

from methods import train_generative as T  # noqa: E402

if __name__ == "__main__":
    T.main()
