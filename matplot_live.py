import random
import time
from itertools import count
from multiprocessing import Process, Queue

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('fivethirtyeight')

def plot_realtime(q):
    x_vals = []
    y_vals = []
    def _update_plot(i):
        while not q.empty():
            x, y = q.get()
            x_vals.append(x)
            y_vals.append(y)
        plt.cla()
        plt.plot(x_vals, y_vals, label='<placeholder>')
        plt.tight_layout()

    # plt.legend(loc='upper left')
    ani = FuncAnimation(plt.gcf(), _update_plot, interval=2000)
    
    plt.tight_layout()
    plt.show()



# ani = FuncAnimation(plt.gcf(), animate, interval=1000)

# plt.tight_layout()
# plt.show()

def main():
    q = Queue()

    animate_proc = Process(target=plot_realtime, args=(q,), daemon=True)
    animate_proc.start()

    for i in range(1, 100):
        x = i
        if i % 2 == 0:
            y = i
        else:
            y = -i
        q.put((x, y))
        time.sleep(0.5)

    animate_proc.join()

if __name__ == '__main__':
    main()