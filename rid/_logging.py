"""Logging and warning-suppression helpers shared by the rid package and experiment scripts.

The library itself only ever logs through ``logging.getLogger("rid...")`` and
installs a ``NullHandler``, so importing ``rid`` stays silent unless the caller
opts in. Experiment entry points call :func:`configure_logging` (timestamped,
flushed-per-line output to stdout, so SLURM ``.out`` files update in real time
instead of only when Python's block buffer fills) and :func:`silence_warnings`.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import warnings

logger = logging.getLogger("rid")
logger.addHandler(logging.NullHandler())

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level="INFO", log_file=None):
    """Route all loggers (``rid`` and the calling script's) to stdout, and optionally a file.

    ``logging.StreamHandler`` flushes after every record, so progress lines reach
    a redirected stdout immediately even without ``python -u``.
    """

    if isinstance(level, str):
        level = getattr(logging, level.upper())

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=handlers,
        force=True,
    )


def add_logging_arguments(parser):
    """Add the shared ``--log-level``/``--log-file`` flags to an experiment script's parser."""

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. DEBUG adds one line per bootstrap (timing, fits, Rashomon size).",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Also write log lines to this file (in addition to stdout).",
    )
    return parser


def silence_warnings():
    """Drop every Python warning in this process and in any worker it spawns later.

    joblib's loky workers are separate processes that do not share this
    process's warning filters, but they do inherit its environment -- so
    ``PYTHONWARNINGS`` must be set before the first ``Parallel`` call starts
    the worker pool. ``compute_rid``/``compute_rid_cross_family`` also ignore
    warnings inside each bootstrap on their own (``suppress_warnings=True``),
    which covers workers that were started before this was called.
    """

    os.environ["PYTHONWARNINGS"] = "ignore"
    warnings.simplefilter("ignore")


def format_duration(seconds):
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


class ProgressLogger:
    """Throttled ``done/total, elapsed, ETA`` progress lines for a long loop.

    Logs when completion crosses each ``pct_step`` percent, or when
    ``min_interval`` seconds have passed since the last line, and always on the
    final item -- so a 500-bootstrap run emits ~20 lines, not 500.
    """

    def __init__(self, label, total, log=None, pct_step=5.0, min_interval=60.0):
        self.label = label
        self.total = max(int(total), 0)
        self.log = log or logger
        self.pct_step = pct_step
        self.min_interval = min_interval
        self.done = 0
        self.start = time.perf_counter()
        self._last_time = self.start
        self._last_bucket = 0

    def update(self, n=1, extra=""):
        self.done += n
        now = time.perf_counter()
        pct = 100.0 * self.done / self.total if self.total else 100.0
        bucket = int(pct // self.pct_step) if self.pct_step else 0
        finished = self.done >= self.total
        if not (finished or bucket > self._last_bucket or now - self._last_time >= self.min_interval):
            return
        self._last_bucket = bucket
        self._last_time = now

        elapsed = now - self.start
        eta = elapsed / self.done * (self.total - self.done) if self.done else float("nan")
        message = (
            f"{self.label}: {self.done}/{self.total} ({pct:.0f}%) "
            f"elapsed={format_duration(elapsed)}"
        )
        if not finished:
            message += f" ETA={format_duration(eta)}"
        if extra:
            message += f" {extra}"
        self.log.info(message)
