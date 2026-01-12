import logging

class PetLogger:
    '''
    A custom logger that logs messages and key-value pairs in a formatted table.

    Parameters:
        filename (str): The name of the log file. Defaults to 'PET.log'.
    '''
    def __init__(self, filename=None):

        self.filename = filename if filename else 'PET.log'
        self.ns = 12  # Number of spaces for table formatting

        # Configurate logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s : %(message)s',
            datefmt='%Y-%m-%d│%H:%M:%S',
            handlers=[
                logging.FileHandler(self.filename, mode='w'),
                logging.StreamHandler()
            ]
        )
        self._logger = logging.getLogger(__name__)


    def __call__(self, *args, **kwargs):
        '''
        Log messages or key-value pairs in a formatted table.

        Parameters:
            *args: Positional arguments to log as a single message.
            **kwargs: Keyword arguments to log in a formatted table.

        Example:
            >>> __call__('This is a log message.')
            2024-06-01│12:00:00 :  This is a log message.
            >>> 
            >>> __call__(iteration=1, fun=0.5, step_size=0.1)
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
                    else:
                        values.append(f'{value:^{self.ns}.3e}')
                except: 
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
        self._logger.info(*args, **kwargs)

    def _set_ns(self, **kwargs):
        '''
        Adjust the number of spaces for table formatting based on the length of keys and values.

        Parameters:
            **kwargs: Keyword arguments to consider for adjusting the space width.
        '''
        for key, value in kwargs.items():
            try:
                if (len(key) > self.ns) or (len(f'{value:.3e}') > self.ns):
                    self.ns = max(len(key), len(f'{value:.3e}')) + 2
            except:
                if len(key) > self.ns:
                    self.ns = len(key) + 2