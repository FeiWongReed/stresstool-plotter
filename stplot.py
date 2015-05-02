#!/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as stats
#import pylab as pl
import matplotlib.pyplot as plt
import zipfile
import shutil
import os
import logging
from datetime import datetime

logger = logging.getLogger('stplot')

def nanos_to_millis(nanos):
    return nanos * 10 ** -6

def nanos_to_ts(nanos):
    return datetime.fromtimestamp(int(nanos) // 1000000000)

def save_figure(output_directory, format, *args):
    name = format % args
    fname = os.path.join(output_directory, name)
    plt.savefig(fname)
    return '<p><img src="%s" alt="%s"/></p>' % (name, name)


def process_zip_report(fname, output_directory):
    zip = zipfile.ZipFile(fname)

    names = zip.namelist()

    zipname = os.path.split(fname)[1]
    success_history = []

    for n in names:
        parts = n.split(os.sep)
        logger.info("Processing %s::%s" % (zipname, n))

        if parts[1] == "calls_successful":
            hostname = parts[0]
            testname = os.path.splitext(parts[2])[0]

            content = zip.open(n, 'r')
            result = np.genfromtxt(content, delimiter=";")

            history = np.delete(result, [0,1,3,5], 1)
            history = np.sort(history, axis=0)
            success_history.append((hostname, testname, history))

    html = "<h2>%s</h2>" % zipname

    for (hostname, testname, history) in success_history:
            timestamps_nanos = history[:, 0]
            result_times = history[:, 1]

            result_msec = np.apply_along_axis(nanos_to_millis, 0, result_times)

            #to_ts = np.vectorize(nanos_to_ts)
            timestamps_msec = np.apply_along_axis(lambda n: nanos_to_millis(n - timestamps_nanos[0]), 0, timestamps_nanos)

            hist = sorted(result_msec)
            fit = stats.norm.pdf(hist, np.mean(hist), np.std(hist))

            html += "<h3>%s at %s</h3>" % (testname, hostname)
            plt.close()
            weights = np.ones_like(hist) / len(hist)
            plt.hist(hist,  histtype='stepfilled', normed=0, weights=weights, cumulative=True, color='b', label='Cumulative probability')
            plt.hist(hist,  histtype='stepfilled', normed=0, weights=weights, color='r', alpha=0.5, label='Probability')
            plt.plot(hist, fit, '-o', color='y')
            plt.title("Normalized distribution of '%s' on %s" % (testname, hostname))
            plt.xlabel("Action time, msec")
            plt.ylabel("Probability")
            plt.legend()
            html += save_figure(output_directory, "%s_%s_dist_normed.png", hostname, testname)

            plt.close()
            plt.hist(hist, bins=30, histtype='stepfilled', normed=False, color='b', label='Frequency')
            plt.title("Distribution of '%s' on %s" % (testname, hostname))
            plt.xlabel("Action time, msec")
            plt.ylabel("Frequency")
            plt.legend()
            html += save_figure(output_directory, "%s_%s_dist_freq.png", hostname, testname)

            plt.close()
            plt.plot(timestamps_msec, result_msec, 'g-^')
            plt.gcf().autofmt_xdate()
            plt.title("History of '%s' on %s" % (testname, hostname))
            plt.xlabel("Time since test start, msec")
            plt.ylabel("Action time, msec")
            plt.legend()
            html += save_figure(output_directory, "%s_%s_history.png", hostname, testname)
    return html


def setup_logging():
    logger = logging.getLogger('stplot')
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


if __name__ == '__main__':
    fname = "sample/stresstest-latitude-rw-2015-121-05--11-49-47.zip"
    output = "output"

    try:
        shutil.rmtree(output)
    except Exception:
        pass
    try:
        os.makedirs(output)
    except Exception:
        pass

    setup_logging()

    with open(os.path.join(output, "report.html"), "w+") as out:
        out.write(process_zip_report(fname, output))