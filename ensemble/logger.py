import logging

class PetLogger:
    '''
    A custom logger that logs messages and key-value pairs in a formatted table.

    Parameters:
        filename (str): The name of the log file. Defaults to 'PET.log'.
    '''
    def __init__(self, filename=None):

        self.filename = filename if filename else 'PET.log'
        self.ns = 10  # Number of spaces for table formatting

        # Configurate logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s : %(levelname)s : %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
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
        '''
        
        if args:
            # Log message from args
            self._logger.info('')
            msg = ' ' + ' '.join(str(arg) for arg in args)
            self._logger.info(msg)

        if kwargs:    
            # Make strings for table logging
            header_parts = []
            values_parts = []
            for key, value in kwargs.items():
                
                # Make sure the ns is large enough
                try:
                    if (len(key) > self.ns) or (len(f'{value:.2e}') > self.ns):
                        self.ns = max(len(key), len(f'{value:.2e}')) + 2
                except:
                    if len(key) > self.ns:
                        self.ns = len(key) + 2

                header_parts.append(f'{key:^{self.ns}}')
                try:
                    if isinstance(value, int) or isinstance(value, str):
                        values_parts.append(f'{value:^{self.ns}}')
                    else:
                        values_parts.append(f'{value:^{self.ns}.2e}')
                except: 
                    values_parts.append(f'{"":^{self.ns}}')
      
            # Log table
            self._logger.info('')
            self._logger.info('  ' + '─' * (len(kwargs) * self.ns + (len(kwargs) - 1) * 3))
            self._logger.info(' │' + ' │ '.join(header_parts) + '│')
            self._logger.info(' │' + '─│─'.join(['─' * self.ns for _ in kwargs.keys()]) + '│')
            self._logger.info(' │' + ' │ '.join(values_parts) + '│')
            self._logger.info('  ' + '─' * (len(kwargs) * self.ns + (len(kwargs) - 1) * 3))