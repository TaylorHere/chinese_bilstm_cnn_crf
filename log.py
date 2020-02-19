import logging


def setUpLogger(trainPath):
    logging.basicConfig(filename=trainPath.info_log_path, level=logging.INFO)
    logging.basicConfig(filename=trainPath.warning_log_path, level=logging.WARNING)
    logging.basicConfig(filename=trainPath.debug_log_path, level=logging.DEBUG)
    logging.basicConfig(filename=trainPath.error_log_path, level=logging.ERROR)
