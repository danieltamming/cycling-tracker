import time
import datetime
import math
from multiprocessing import Process, Queue

import numpy as np
import pandas as pd
from scipy import fftpack
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import cv2


class FreqTracker():
    def __init__(self, plot_q, algo_params, thresh_params, user_params):
        # time.sleep(5)
        print('Warming up...')
        self.plot_q = plot_q
        self.algo_params = algo_params
        self.thresh_params = thresh_params
        self.user_params = user_params
        self.frame_mem = []
        # length <= WINDOW_SIZE always
        self.motion_mem = []
        self.mag_mem = None
        # counts motion frames,
        # of which there are always 2 less than image frames
        self.frame_count = -2
        self.data = []
        self.freqs = None
        self.max_rps = self.thresh_params['MAX_PEDAL_CADENCE'] / 60
        self.mag_history = None
        self.distance = 0

    def run_feed(self):
        # cap = cv2.VideoCapture(0)
        cap = cv2.VideoCapture('../cycling-tracker-private/videos/short_vid3.MOV')
        self.start_time = time.time()
        self.time = self.start_time
        try:
            while(cap.isOpened()):
                ret, frame = cap.read()
                ######################## COMMENT OUT BELOW IF USING VIDEO
                time.sleep(1 / 30)
                if ret:
                    self.process_image(frame)
                else:
                    break
        except KeyboardInterrupt:
            print('Exiting program.')
            cap.release()

    def process_image(self, img):
        img = cv2.resize(img, (400, 400))
        frame = self.filter_image(img)
        # if haven't built up a window size
        if len(self.motion_mem) < self.algo_params['WINDOW_SIZE']:
            if len(self.frame_mem) < 3:
                self.frame_mem += [frame]
            else:
                self.frame_mem = self.frame_mem[-2:] + [frame]
            if len(self.frame_mem) == 3:
                self.motion_mem += [self.get_motion(self.frame_mem)]
        # if window size has been built up
        else:
            # keep last two and add new
            self.frame_mem = self.frame_mem[-2:] + [frame]
            # keep last WINDOW_SIZE-1 and add new
            self.motion_mem = (
                self.motion_mem[-1 * (self.algo_params['WINDOW_SIZE'] - 1):] +
                [self.get_motion(self.frame_mem)]
            )
        self.frame_count += 1
        if (len(self.motion_mem) == self.algo_params['WINDOW_SIZE'] and
                self.frame_count % self.algo_params['REFRESH_PERIOD'] == 0):
            self.update_mag_mem()

    def update_mag_mem(self):
        '''
        Returns absolute difference between final frame in list
        and all those before it
        '''
        # start_time = time.time()

        S_col = np.zeros(len(self.motion_mem))
        for i in range(len(self.motion_mem)):
            S_col[i] = self.get_abs_diff(
                self.motion_mem[i], self.motion_mem[-1])

        if S_col.shape[0] < self.algo_params['WINDOW_SIZE']:
            S_col = np.concatenate(
                (np.zeros(self.algo_params['WINDOW_SIZE'] - S_col.shape[0]),
                    S_col))

        freqs, mags = self.get_freq_spec(
            S_col, self.user_params['FRAME_SPEED'])

        if self.freqs is None:
            self.freqs = freqs

        if self.mag_history is None:
            self.mag_history = mags
        else:
            self.mag_history = np.column_stack((self.mag_history, mags))

        if self.mag_mem is None:
            self.mag_mem = np.zeros_like(mags)
        self.mag_mem = (self.algo_params['MEMORY_BETA'] * self.mag_mem
                        + (1 - self.algo_params['MEMORY_BETA']) * mags)

        self.update_averages()

    def update_averages(self):
        rpm = 60 * self.freqs[np.argmax(self.mag_mem)]
        kmph = (2 * math.pi * self.user_params['WHEEL_RADIUS']
                * self.user_params['WHEELS_PER_PEDAL'] * rpm * 60 / 1000)
        time_delta = time.time() - self.time  # seconds
        self.distance += kmph * time_delta / (60 * 60)
        self.time += time_delta
        self.mean_speed = self.distance / (self.time - self.start_time)
        print(
            'RPM: {}, Frame: {}, Speed (km / h): {},'
            ' Average Speed (km / h): {}, Distance: {} (km),'
            ' Time: {}'.format(
                round(rpm, 3),
                self.frame_count,
                round(kmph, 3),
                round(self.mean_speed, 3),
                round(self.distance, 3),
                str(
                    datetime.timedelta(
                        seconds=self.time-self.start_time
                    )
                ).split('.')[0]
            )
        )
        data_dict = {
            'frame_count': self.frame_count,
            'second': self.time - self.start_time,
            'rpm': rpm,
            'speed': kmph,
            'distance': self.distance
        }

        # self.plot_q.put((self.time - self.start_time, rpm))
        self.plot_q.put(data_dict)

        # self.data.append((self.frame_count, rpm))
        self.data.append(data_dict)

    def get_freq_spec(self, vec, f_s, pos_only=True):
        vec = vec - vec.mean()
        # pad to make vector this large
        N = self.algo_params['INTER_CONST'] * len(vec)
        mags = np.abs(fftpack.fft(vec, n=N))
        freqs = fftpack.fftfreq(N) * f_s
        pos_mask = freqs >= 0
        freqs = freqs[pos_mask]
        mags = mags[pos_mask]
        if self.max_rps is not None:
            valid_mask = freqs <= self.max_rps
            freqs = freqs[valid_mask]
            mags = mags[valid_mask]
        return freqs, mags

    def filter_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(
            gray, self.algo_params['GAUSSIAN_FILTER'], 0)
        return blurred

    def get_motion(self, frame_list):
        assert len(frame_list) == 3
        motion_1, motion_2 = [
            (np.absolute(frame_list[i] - frame_list[i+1])
                > self.thresh_params['MOTION_THRESH'])
            for i in range(len(frame_list) - 1)
        ]
        anded_motion = np.float32(np.logical_and(motion_1, motion_2))
        opened_motion = cv2.morphologyEx(
            anded_motion, cv2.MORPH_OPEN, self.algo_params['MOTION_KERNEL'])
        opened_motion = np.float32(
            cv2.resize(opened_motion, (200, 200)) >= 0.5
        )
        return opened_motion

    def get_abs_diff(self, mot1, mot2):
        return np.absolute(mot2 - mot1).sum()

    # def plot_data(self):
    #     df = pd.DataFrame.from_dict(self.data)
    #     sns.lineplot(data=df x='second', y='rpm')
    #     plt.xlabel('Second')
    #     plt.ylabel('RPM')
    #     plt.show()


def plot_realtime(q):
    second_list = []
    rpm_list = []
    speed_list = []
    distance_list = []
    fig, ax = plt.subplots(2, 2)
    rpm_max = 160
    speed_max = 60
    distance_max = 10
    ax[0, 0].set_ylim([0, rpm_max])
    ax[0, 1].set_ylim([0, speed_max])
    ax[1, 0].set_ylim([0, distance_max])
    def _update_plot(i):
        while not q.empty():
            data_dict = q.get()
            second_list.append(data_dict['second'])
            rpm_list.append(data_dict['rpm'])
            speed_list.append(data_dict['speed'])
            distance_list.append(data_dict['distance'])
        # plt.cla()
        # ax[0, 0].clear()
        ax[0, 0].plot(second_list, rpm_list)
        # ax[0, 1].clear()
        ax[0, 1].plot(second_list, speed_list)
        # ax[1, 0].clear()
        ax[1, 0].plot(second_list, distance_list)
        # plt.plot(x_vals, y_vals, label='<placeholder>')
        # plt.tight_layout()

    # plt.legend(loc='upper left')
    ani = FuncAnimation(fig, _update_plot, interval=250)
    # ani = FuncAnimation(plt.gcf(), _update_plot, interval=250)
    
    # plt.tight_layout()
    plt.show()

def main():
    # Constants that can, but do not have to be, changed by users
    algo_params = dict(
        # Number of frames to keep in memory
        WINDOW_SIZE=256,
        # Vector padding multiple (larger=more granular interpolation)
        INTER_CONST=8,
        # Update RPM reading after this many video frames
        REFRESH_PERIOD=30,
        # Exponentially weighted average smoothing parameter
        MEMORY_BETA=0.80,
        # Image-smoothing filter
        GAUSSIAN_FILTER=(9, 9),
        # Filter for morphological open
        MOTION_KERNEL=np.ones((20, 20))
    )
    # Thresholds that may be specific to the person, bike, and / or setting
    thresh_params = dict(
        # Minimum grayscale change required to be considered motion
        MOTION_THRESH=150,
        # Cadences can't be increased / decreased by more than this factor
        CADENCE_MULTIPLE_MAX=2,
        # Max cadence in RPMs
        MAX_PEDAL_CADENCE=150
    )
    # Constants that must match the camera and bike being used
    user_params = dict(
        # Camera frame speed
        FRAME_SPEED=30,
        # Wheel radius in meters
        WHEEL_RADIUS=0.22,
        # Number of wheel revolutions per pedal revolution
        WHEELS_PER_PEDAL=3.5
    )

    plot_q = Queue()
    plot_proc = Process(target=plot_realtime, args=(plot_q, ), daemon=True)
    pedal_tracker = FreqTracker(
        plot_q, algo_params, thresh_params, user_params)
    plot_proc.start()
    pedal_tracker.run_feed()
    plot_proc.join()


    # pedal_tracker.plot_data()

if __name__ == "__main__":
    main()
