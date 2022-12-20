import numpy as np

def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin):
    r_histo, _ = np.histogram(frame[ymin:ymax, xmin:xmax, 0], bins = hist_bin, range = (0,255))
    g_histo, _ = np.histogram(frame[ymin:ymax, xmin:xmax, 1], bins = hist_bin, range = (0,255))
    b_histo, _ = np.histogram(frame[ymin:ymax, xmin:xmax, 2], bins = hist_bin, range = (0,255))

    hist = np.concatenate([r_histo, g_histo, b_histo])
    hist = hist/np.sum(hist)
    return hist
