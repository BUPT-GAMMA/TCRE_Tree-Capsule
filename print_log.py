import sys
import os
from tqdm import tqdm
import time
import torch

class Logger(object):
    def __init__(self, filename="Default.log", remove=True):
        self.terminal = sys.stdout
        if filename[-4:] != '.log': filename += '.log'
        if os.path.exists(filename) and remove:
            os.remove(filename)
        self.log = open(filename, "a")
        t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.log.write("\n\n{0} {1} {0}\n".format('-'*60, t))

    def write(self, message):
        self.terminal.write(message)
        # if 'Iter' not in message:
        self.log.write(message)


    def flush(self):
        pass


def occumpy_mem(memory):
    block_mem = int(memory * 1000000 / 1024 / 1024)
    x = []; i = 0
    with tqdm(total=(block_mem // 200)) as t:
        while i < (block_mem // 200):
            try:
                x.append(torch.cuda.FloatTensor(256, 1024, 200))
                time.sleep(0.06)
                i += 1
                t.update()
            except:
                time.sleep(5)
    return x

if __name__ == '__main__':
    sys.stdout = Logger("yourlogfilename.txt")
    print('content.')