#!/usr/bin/env python
from __future__ import print_function
import cv2
import numpy as np
eps = 1e-5

def rnd_warp(a):
    h, w = a.shape[:2]
    T = np.zeros((2, 3))
    coef = 0.2
    ang = (np.random.rand()-0.5)*coef
    c, s = np.cos(ang), np.sin(ang)
    T[:2, :2] = [[c,-s], [s, c]]
    T[:2, :2] += (np.random.rand(2, 2) - 0.5)*coef
    c = (w/2, h/2)
    T[:,2] = c - np.dot(T[:2, :2], c)
    return cv2.warpAffine(a, T, (w, h), borderMode = cv2.BORDER_REFLECT)

def divSpec(A, B):
    Ar, Ai = A[...,0], A[...,1]
    Br, Bi = B[...,0], B[...,1]
    C = (Ar+1j*Ai)/(Br+1j*Bi)
    C = np.dstack([np.real(C), np.imag(C)]).copy()
    return C

def draw_str(dst, target, s):
    x, y = target

class MOSSE:
    def __init__(self, frame, rect):
        x1, y1, x2, y2 = rect
        w, h = map(cv2.getOptimalDFTSize, [x2-x1, y2-y1])
        x1, y1 = (x1+x2-w)//2, (y1+y2-h)//2
        self.pos = x, y = x1+0.5*(w-1), y1+0.5*(h-1)
        self.size = w, h
        img = cv2.getRectSubPix(frame, (w, h), (x, y))

        self.win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        g = np.zeros((h, w), np.float32)
        g[h//2, w//2] = 1
        g = cv2.GaussianBlur(g, (-1, -1), 2.0)
        g /= g.max()

        self.G = cv2.dft(g, flags=cv2.DFT_COMPLEX_OUTPUT)
        self.H1 = np.zeros_like(self.G)
        self.H2 = np.zeros_like(self.G)
        for i in xrange(128):
            a = self.preprocess(rnd_warp(img))
            A = cv2.dft(a, flags=cv2.DFT_COMPLEX_OUTPUT)
            self.H1 += cv2.mulSpectrums(self.G, A, 0, conjB=True)
            self.H2 += cv2.mulSpectrums(     A, A, 0, conjB=True)
        self.update_kernel()
        self.update(frame)

    def update(self, frame, rate = 0.125):
        (x, y), (w, h) = self.pos, self.size
        self.last_img = img = cv2.getRectSubPix(frame, (w, h), (x, y))
        img = self.preprocess(img)
        self.last_resp, (dx, dy), self.psr = self.correlate(img)
        self.good = self.psr > 8.0
        if not self.good:
            return
        self.pos = x+dx, y+dy
        self.last_img = img = cv2.getRectSubPix(frame, (w, h), self.pos)
        img = self.preprocess(img)

        A = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
        H1 = cv2.mulSpectrums(self.G, A, 0, conjB=True)
        H2 = cv2.mulSpectrums(     A, A, 0, conjB=True)
        self.H1 = self.H1 * (1.0-rate) + H1 * rate
        self.H2 = self.H2 * (1.0-rate) + H2 * rate
        self.update_kernel()

    # @property
    def state_vis(self):
        f = cv2.idft(self.H, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
        h, w = f.shape
        f = np.roll(f, -h//2, 0)
        f = np.roll(f, -w//2, 1)
        kernel = np.uint8( (f-f.min()) / f.ptp()*255 )
        resp = self.last_resp
        resp = np.uint8(np.clip(resp/resp.max(), 0, 1)*255)
        vis = np.hstack([self.last_img, kernel, resp])
        return vis

    def draw_state(self, vis):
        (x, y), (w, h) = self.pos, self.size
        x1, y1, x2, y2 = int(x-0.5*w), int(y-0.5*h), int(x+0.5*w), int(y+0.5*h)
        #cv2.rectangle(vis, (x1, y1), (x2, y2), (50, 200, 50), 5)
        cv2.line(vis, (x1, y1), (x1, y1+16), (50, 200, 50), 2)
        cv2.line(vis, (x1, y1), (x1+16, y1), (50, 200, 50), 2)
        cv2.line(vis, (x2, y1), (x2, y1+16), (50, 200, 50), 2)
        cv2.line(vis, (x2, y1), (x2-16, y1), (50, 200, 50), 2)
        cv2.line(vis, (x1, y2), (x1, y2-16), (50, 200, 50), 2)
        cv2.line(vis, (x1, y2), (x1+16, y2), (50, 200, 50), 2)
        cv2.line(vis, (x2, y2), (x2-16, y2), (50, 200, 50), 2)
        cv2.line(vis, (x2, y2), (x2, y2-16), (50, 200, 50), 2)
        if self.good:
            #cv2.circle(vis, (int(x), int(y)), 2, (50, 200, 50), -1)
            cv2.line(vis, (int(x)-4, int(y)), (int(x)+4, int(y)), (50, 200, 50), 2)
            cv2.line(vis, (int(x), int(y) - 4), (int(x), int(y)+4), (50, 200, 50), 2)
        else:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
            cv2.line(vis, (x2, y1), (x1, y2), (0, 0, 255))
        draw_str(vis, (x1, y2+16), 'PSR: %.2f' % self.psr)

    def preprocess(self, img):
        img = np.log(np.float32(img)+1.0)
        img = (img-img.mean()) / (img.std()+eps)
        return img*self.win

    def correlate(self, img):
        C = cv2.mulSpectrums(cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT), self.H, 0, conjB=True)
        resp = cv2.idft(C, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        h, w = resp.shape
        _, mval, _, (mx, my) = cv2.minMaxLoc(resp)
        side_resp = resp.copy()
        cv2.rectangle(side_resp, (mx-5, my-5), (mx+5, my+5), 0, -1)
        smean, sstd = side_resp.mean(), side_resp.std()
        psr = (mval-smean) / (sstd+eps)
        return resp, (mx-w//2, my-h//2), psr

    def update_kernel(self):
        self.H = divSpec(self.H1, self.H2)
        self.H[...,1] *= -1
