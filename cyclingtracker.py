import os
import time
import datetime
import math

from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import signal
from scipy import fftpack
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

GAUSSIAN_FILTER = (9, 9)
MOTION_THRESH = 150
MOTION_KERNEL = np.ones((20, 20))
# length is WINDOW_SIZE/FRAME SPEED seconds
FRAME_SPEED = 30
WINDOW_SIZE = 256
# pad vec to make it INTER_CONST times longer
# larger values mean more granular interpolation
INTER_CONST = 8
# revolutions per minute resolution before interpolation
RESOLUTION = 60 * FRAME_SPEED / WINDOW_SIZE
# max cadence in RPMs
MAX_PEDAL_CADENCE = 150
WHEEL_RADIUS = 0.22
# get new S_col every REFRESH_PERIOD frames
REFRESH_PERIOD = 30
# memory is approx 1 / (1 - MEMORY_BETA)
MEMORY_BETA = 0.80
# wheel has WHEELS_PER_PEDAL revolutions per pedal revolution
WHEELS_PER_PEDAL = 3.5
MAX_WHEEL_CADENCE = MAX_PEDAL_CADENCE * WHEELS_PER_PEDAL
# filter cadences so they can't be increased/decreased by more than this factor
CADENCE_MULTIPLE_MAX = 2


class FreqTracker():
    def __init__(self):
        time.sleep(2)
        print('Warming up...')
        self.frame_mem = []
        # length <= WINDOW_SIZE always
        self.motion_mem = []
        self.mag_mem = None
        # counts motion frames,
        # of which there are always 2 less than image frames
        self.frame_count = -2
        self.data = []
        self.freqs = None
        self.max_rps = MAX_PEDAL_CADENCE/60
        self.mag_history = None
        self.distance = 0

    def run_feed(self):
        cap = cv2.VideoCapture(0)
        self.start_time = time.time()
        self.time = self.start_time
        try:
            while(cap.isOpened()):
                ret, frame = cap.read()
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
        if len(self.motion_mem) < WINDOW_SIZE:
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
                self.motion_mem[-1*(WINDOW_SIZE - 1):] +
                [self.get_motion(self.frame_mem)]
            )
        self.frame_count += 1
        if (len(self.motion_mem) == WINDOW_SIZE and
                self.frame_count % REFRESH_PERIOD == 0):
            self.update_mag_mem()

    def update_mag_mem(self):
        '''
        Returns absolute difference between final frame in list
        and all those before it
        '''
        start_time = time.time()

        S_col = np.zeros(len(self.motion_mem))
        for i in range(len(self.motion_mem)):
            S_col[i] = self.get_abs_diff(
                self.motion_mem[i], self.motion_mem[-1])

        if S_col.shape[0] < WINDOW_SIZE:
            S_col = np.concatenate(
                (np.zeros(WINDOW_SIZE - S_col.shape[0]), S_col))

        freqs, mags = self.get_freq_spec(S_col, FRAME_SPEED)

        if self.freqs is None:
            self.freqs = freqs

        if self.mag_history is None:
            self.mag_history = mags
        else:
            self.mag_history = np.column_stack((self.mag_history, mags))

        if self.mag_mem is None:
            self.mag_mem = np.zeros_like(mags)
        self.mag_mem = MEMORY_BETA*self.mag_mem + (1 - MEMORY_BETA)*mags

        self.update_averages()

    def update_averages(self):
        rpm = 60*self.freqs[np.argmax(self.mag_mem)]
        kmph = 2*math.pi*WHEEL_RADIUS*WHEELS_PER_PEDAL*rpm*60/1000
        time_delta = time.time() - self.time  # seconds
        self.distance += kmph*time_delta/(60*60)
        self.time += time_delta
        self.mean_speed = self.distance / (self.time - self.start_time)
        print(
            'RPM: {}, Frame: {}, Speed (km/h): {},'
            ' Average Speed (km/h): {}, Distance: {} (km),'
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
        self.data.append((self.frame_count, rpm))

    def get_freq_spec(self, vec, f_s, pos_only=True):
        vec = vec - vec.mean()
        N = INTER_CONST*len(vec)  # how large to make vec after padding
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
        blurred = cv2.GaussianBlur(gray, GAUSSIAN_FILTER, 0)
        return blurred

    def get_motion(self, frame_list):
        assert len(frame_list) == 3
        motion_1, motion_2 = [
            np.absolute(frame_list[i] - frame_list[i+1]) > MOTION_THRESH
            for i in range(len(frame_list) - 1)
        ]
        anded_motion = np.float32(np.logical_and(motion_1, motion_2))
        opened_motion = cv2.morphologyEx(
            anded_motion, cv2.MORPH_OPEN, MOTION_KERNEL)
        opened_motion = np.float32(
            cv2.resize(opened_motion, (200, 200)) >= 0.5
        )
        return opened_motion

    def get_abs_diff(self, mot1, mot2):
        return np.absolute(mot2 - mot1).sum()

    def get_data(self):
        df = pd.DataFrame(self.data, columns=['frame_num', 'rpm'])
        df['second'] = df['frame_num'] / FRAME_SPEED
        return df

    def plot_data(self):
        pedal_data = self.get_data()
        sns.lineplot(data=pedal_data, x='second', y='rpm')
        plt.xlabel('Second')
        plt.ylabel('RPM')
        plt.show()


pedal_tracker = FreqTracker()
pedal_tracker.run_feed()
pedal_tracker.plot_data()