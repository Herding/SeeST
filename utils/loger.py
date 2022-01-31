import logging


# https://www.cnblogs.com/qianyuliang/p/7234217.html
LEVER = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

class Loger():
    def __init__(self,
                 model_name,
                 log_path,
                 lever='INFO',
                 fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                 datafmt="%Y/%m/%d %H:%M"
                 ):
        
        file_name = log_path + model_name + '.txt'
        
        fmter = logging.Formatter(fmt=fmt, datefmt=datafmt)
        
        sh = logging.FileHandler(filename=file_name)
        sh.setLevel(LEVER[lever])
        sh.setFormatter(fmter)

        console = logging.StreamHandler()
        console.setLevel(LEVER[lever])
        console.setFormatter(fmter)

        self._loger = logging.getLogger(model_name)
        self._loger.setLevel(LEVER[lever])
        self._loger.addHandler(sh)
        self._loger.addHandler(console)

    @property
    def loger(self):
        return self._loger
    
    def add_info(self, information, lever):
        if lever == 'DEBUG':
            self._loger.debug(information)
        elif lever == 'INFO':
            self._loger.info(information)
        elif lever == 'WARNING':
            self._loger.warning(information)
        elif lever == 'ERROR':
            self._loger.error(information)
        elif lever == 'CRITICAL':
            self._loger.critical(information)
        else:
            raise ValueError(f'{lever} is not defined in [DEBUG, INFO, WARNING, ERROR, CRITICAL]')
