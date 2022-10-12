import os
import time
from multiprocessing import Process, Queue
import cv2


class WebCamControl:
    def __init__(self, cam_id=0):
        capture_size   = (1920, 1080)
        # capture_size   = (3840, 2160)
        self.img_size  = (1000, 1000)
        self.w, self.h = capture_size[0], capture_size[1]
        self.fps       = 30  # default
        self.video_len = 1  # [s]

        self.cam = cv2.VideoCapture(cam_id)
        self.cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cam.set(3, capture_size[0])
        self.cam.set(4, capture_size[1])

        self.fname_queue, self.stop_queue = Queue(), Queue()
        self.p = Process(target=self.record_process, args=(self.fname_queue, self.stop_queue))
        self.p.start()

    def record_process(self, fname_queue, stop_queue):
        """
        :param fname_queue: dir_name + fname (e.g. /hoge/fuga/piyo.mp4)
        :param stop_queue: break_flag
        :return:
        """
        w, h = self.w, self.h
        # w, h = self.img_size
        # h = 800

        while True:
            fname = fname_queue.get()
            if fname == 'FIN':
                print('camera_process has been finished.')
                break
            out = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), self.fps, (w, h))
            imgs = []
            while True:
                _, frame = self.cam.read()
                imgs.append(frame.copy())
                print(imgs[0].shape)
                try:
                    fl = stop_queue.get_nowait()
                except:
                    continue
                if fl:
                    break
            # [out.write(e[int((self.h - h)/2):-int((self.h - h)/2), int((self.w - w)/2):-int((self.w - w)/2)]) for e in imgs]
            # [out.write(e[int((self.h - 1000)/2)+125:-int((self.h - 1000)/2)-75, int((self.w - w)/2):-int((self.w - w)/2)]) for e in imgs]
            [out.write(e) for e in imgs]
            out.release()
            print('created')

        self.cam.release()
        print('camera object is released!')

    def rec_start(self, fname: str):
        """
        Record start
        :param fname_queue: dir_name + fname (e.g. /hoge/fuga/piyo.mp4)
        """
        self.fname_queue.put(fname)

    def rec_stop(self):
        """
        Record stop
        """
        self.stop_queue.put(True)

    def release(self):
        """
        Cam object relase
        """
        self.fname_queue.put('FIN')


if __name__ == '__main__':
    webcam = WebCamControl(2)

    for h in range(1,2):
        # rec start
        webcam.rec_start(fname=f'/home/tomoya-y/workspace/test_video{h}.mp4')
        # Do somthing...
        time.sleep(h)
        # Do somthing...
        webcam.rec_stop()
    # Release
    webcam.release()
