import logging
import time

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s  %(message)s', datefmt='%Y/%m/%d %H:%M:%S')

# 使用FileHandler输出到文件
now_time = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
fh = logging.FileHandler(f'{now_time}.log', mode='w')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)

# 使用StreamHandler输出到屏幕
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)

# 添加两个Handler
logger.addHandler(ch)
logger.addHandler(fh)
