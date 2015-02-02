import numpy as np


def auto_bins(vmin, vmax, nbins=10):
    steps = [0.5, 1, 1.5, 2, 2.5, 4, 5, 6, 8, 10]
    scale, offset = scale_range(vmin, vmax, nbins)
    vmin -= offset
    vmax -= offset
    raw_step = (vmax-vmin)/nbins
    scaled_raw_step = raw_step/scale
    best_vmax = vmax
    best_vmin = vmin

    for step in steps:
        if step < scaled_raw_step:
            continue
        step *= scale
        best_vmin = step*divmod(vmin, step)[0]
        best_vmax = best_vmin + step*nbins
        if (best_vmax >= vmax):
            break
    return (np.arange(nbins+1) * step + best_vmin + offset)


def scale_range(vmin, vmax, n=1, threshold=100):
    dv = abs(vmax - vmin)
    maxabsv = max(abs(vmin), abs(vmax))
    if maxabsv == 0 or dv/maxabsv < 1e-12:
        return 1.0, 0.0
    meanv = 0.5*(vmax+vmin)
    if abs(meanv)/dv < threshold:
        offset = 0
    elif meanv > 0:
        ex = divmod(np.log10(meanv), 1)[0]
        offset = 10**ex
    else:
        ex = divmod(np.log10(-meanv), 1)[0]
        offset = -10**ex
    ex = divmod(np.log10(dv/n), 1)[0]
    scale = 10**ex
    return scale, offset
