"""Run logging: a table-formatting file logger, a no-op stand-in for when logging is off, and
helpers that format the lines of a detailed log."""
import logging

import numpy as np

__all__ = ["PetLogger", "NullLogger", "block", "change", "format_duration", "format_value", "heading", "items"]


class NullLogger:
    """Callable no-op standing in for a :class:`PetLogger` when logging is
    disabled -- so callers can invoke ``self.logger(...)`` unconditionally
    without checking whether logging is on, and no log file is created.
    """

    def __call__(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        """No-op."""
        pass

class PetLogger:
    '''
    A custom logger that logs messages and key-value pairs in a formatted table.

    Parameters:
        filename (str): The name of the log file. Defaults to 'PET.log'.
    '''
    def __init__(self, filename=None):

        self.filename = filename if filename else 'PET.log'
        self.ns = 12  # Number of spaces for table formatting

        # One named logger per log file, carrying its own file and console
        # handlers. This used to call logging.basicConfig, which configures
        # the *root* logger once per process and silently does nothing the
        # second time -- so a second PetLogger (popt beside pipt, or a re-run
        # in a notebook) kept writing into the first file, and any test or
        # application that had touched the root logger got no file at all.
        # Records still propagate upward, so a root handler (pytest's capture,
        # an application's own configuration) sees them too.
        self._logger = logging.getLogger(f"pet.{self.filename}")
        self._logger.setLevel(logging.INFO)
        for handler in list(self._logger.handlers):
            self._logger.removeHandler(handler)
            handler.close()
        formatter = logging.Formatter('%(asctime)s : %(message)s', datefmt='%Y-%m-%d│%H:%M:%S')
        for handler in (logging.FileHandler(self.filename, mode='w', encoding='utf-8'), logging.StreamHandler()):
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)


    def __call__(self, *args, **kwargs):
        '''
        Log messages or key-value pairs in a formatted table.

        Parameters:
            *args: Positional arguments to log as a single message.
            **kwargs: Keyword arguments to log in a formatted table.

        Example:
            >>> logger = PetLogger()
            >>> logger('This is a log message.')
            2024-06-01│12:00:00 :  This is a log message.
            >>>
            >>> logger(iteration=1, fun=0.5, step_size=0.1)
            2024-06-01│12:00:00 :
            2024-06-01│12:00:00 : ┌────────────┬────────────┬────────────┐
            2024-06-01│12:00:00 : │ iteration  │    fun     │ step_size  │
            2024-06-01│12:00:00 : ├────────────┼────────────┼────────────┤
            2024-06-01│12:00:00 : │     1      │  5.000e-01 │  1.000e-01 │
            2024-06-01│12:00:00 : └────────────┴────────────┴────────────┘
            2024-06-01│12:00:00 :
        '''

        if args:
            # Log message from args
            msg = ' ' + ' '.join(str(arg) for arg in args)
            self._logger.info(msg)

        if kwargs:
            # Make strings for table logging
            self._set_ns(**kwargs)
            header = []
            values = []
            for key, value in kwargs.items():
                header.append(f'{key:^{self.ns}}')
                try:
                    if isinstance(value, int) or isinstance(value, str):
                        values.append(f'{value:^{self.ns}}')
                    elif '%' in key:
                        values.append(f'{value:^{self.ns}.2f}')
                    else:
                        values.append(f'{value:^{self.ns}.3e}')
                except Exception:
                    values.append(f'{"":^{self.ns}}')

            # Log table
            seperator = ['─' * self.ns for _ in kwargs.keys()]
            self._logger.info('')
            self._logger.info(' ┌' + '┬'.join(seperator) + '┐')
            self._logger.info(' │' + '│'.join(header)    + '│')
            self._logger.info(' ├' + '┼'.join(seperator) + '┤')
            self._logger.info(' │' + '│'.join(values)    + '│')
            self._logger.info(' └' + '┴'.join(seperator) + '┘')
            self._logger.info('')

    def info(self, *args, **kwargs):
        """Log as given; ``__call__`` is the table-aware form."""
        self._logger.info(*args, **kwargs)

    def _set_ns(self, **kwargs):
        '''
        Adjust the number of spaces for table formatting based on the length of keys and values.

        Parameters:
            **kwargs: Keyword arguments to consider for adjusting the space width.
        '''
        self.ns = 12
        for key, value in kwargs.items():
            value_len = 0
            try:
                if isinstance(value, int) or isinstance(value, str):
                    value_len = len(str(value))
                elif '%' in key:
                    value_len = len(f'{value:.2f}')
                else:
                    value_len = len(f'{value:.3e}')
            except Exception:
                value_len = 0

            self.ns = max(self.ns, len(key) + 2, value_len + 2)


# ---------------------------------------------------------------------- #
# Lines of a detailed log
# ---------------------------------------------------------------------- #
# Each helper returns text, one line or a list of lines, and logs nothing: a
# scheme hands the lines to its logger one by one. Together they give a log
# with a block for the problem, a heading for every iteration and under it
# short lists of what was computed, for example
#
#     ── Iteration 3  (12 min 05 s) ──────────────────────────────────────────
#     Kernel
#        Exponent p     1.82
#        Bandwidth σ    21.3
RULE_WIDTH = 72


def format_value(value) -> str:
    """Short text for a value. Arrays, lists and tuples are described by their size, never by their elements."""
    if isinstance(value, (bool, np.bool_)) or value is None:
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        return format_value(value.item()) if value.ndim == 0 else f"array of shape {value.shape}"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__} of length {len(value)}"
    if callable(value):
        return getattr(value, "__name__", type(value).__name__)
    return type(value).__name__


def heading(title: str) -> str:
    """A rule with a title, e.g. ``── Iteration 3 ─────``."""
    text = f"── {title} "
    return text + "─" * max(RULE_WIDTH - len(text), 3)


def block(title: str, rows, heavy: bool = False) -> list:
    """The lines of a titled block of aligned rows, between rules.

    Parameters
    ----------
    title : str
        Title of the block.
    rows : sequence of tuples
        ``(label, value)`` or ``(label, symbol, value)``. A value that is not
        a string is formatted with :func:`format_value`.
    heavy : bool
        Double rules, and one below the rows: for the start and the end of a run.
    """
    rule = ("═" if heavy else "─") * RULE_WIDTH
    rows = [(row[0], "", row[1]) if len(row) == 2 else tuple(row) for row in rows]
    label_width = max([len(label) for label, _, _ in rows], default=0)
    symbol_width = max([len(symbol) for _, symbol, _ in rows], default=0)
    lines = [rule, f" {title}", rule]
    for label, symbol, value in rows:
        text = value if isinstance(value, str) else format_value(value)
        symbol = f"{symbol:<{symbol_width}}  " if symbol_width else ""
        lines.append(f"   {label:<{label_width}}  {symbol}{text}")
    if heavy:
        lines.append(rule)
    return lines


def items(title, rows, width=None) -> list:
    """The lines of a titled list with one value per line.

    Parameters
    ----------
    title : str or None
        The line above the list; ``None`` continues a list begun earlier.
    rows : sequence of tuples
        ``(label, value)``. A value that is not a string is formatted with
        :func:`format_value`.
    width : int, optional
        Width of the labels, so that a list written in two parts lines up.
        Default: that of the longest label.
    """
    if width is None:
        width = max([len(label) for label, _ in rows], default=0)
    lines = [] if title is None else [title]
    for label, value in rows:
        text = value if isinstance(value, str) else format_value(value)
        lines.append(f"   {label:<{width}}  {text}")
    return lines


def change(name: str, old, new, note: str = "", fmt: str = ".3e") -> str:
    """One line for a quantity that changed, e.g. ``↳ λ  100 → 10  (step accepted)``."""
    note = f"  ({note})" if note else ""
    return f"   ↳ {name}  {old:{fmt}} → {new:{fmt}}{note}"


def format_duration(seconds) -> str:
    """Wall time as ``45.2 s``, ``12 min 05 s`` or ``3 h 07 min``."""
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, seconds = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes} min {seconds:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"
