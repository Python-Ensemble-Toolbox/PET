"""Run logging: a table-formatting file logger and a no-op stand-in for when logging is off."""
import logging

__all__ = ["PetLogger", "NullLogger"]


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
        for handler in (logging.FileHandler(self.filename, mode='w'), logging.StreamHandler()):
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
