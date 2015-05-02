#!/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as stats
#import pylab as pl
import matplotlib.pyplot as plt
import zipfile
import shutil
import os
import sys
import logging
from datetime import datetime
import tempfile

logger = logging.getLogger('stplot')

def nanos_to_millis(nanos):
    return nanos * 10 ** -6

def nanos_to_ts(nanos):
    return datetime.fromtimestamp(int(nanos) // 1000000000)

def save_figure(output_directory, format, *args):
    name = format % args
    fname = os.path.join(output_directory, name)
    plt.savefig(fname)
    return '<img src="%s" alt="%s"/>' % (name, name)

def zipdir(path, zip):
    for root, dirs, files in os.walk(path):
        for file in files:
            zip.write(os.path.join(root, file), arcname=file)


class Test(object):
    def __init__(self, hostname, testname, success_history, errors):
        self.hostname = hostname
        self.testname = testname
        self.success_history = success_history
        self.errors = errors

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

            with zip.open(n, 'r') as content:
                result = np.genfromtxt(content, delimiter=";")
                history = np.delete(result, [0,1,3,5], 1)
                history = np.sort(history, axis=0)

                err_parts = parts
                err_parts[1] = "calls_failed"
                errs = None
                errname = os.sep.join(err_parts)
                if errname in names:
                    with zip.open(errname) as err_content:
                        err_result = np.genfromtxt(err_content, delimiter=";")
                        errs = np.delete(err_result, [0,1,3,5,6], 1)
                        errs = np.sort(errs, axis=0)
                success_history.append(Test(hostname, testname, history, errs))

    html = "<h2>%s</h2>" % zipname

    for test in success_history:
            hostname = test.hostname
            testname = test.testname
            history = test.success_history

            timestamps_nanos = history[:, 0]
            result_times = history[:, 1]

            result_msec = np.apply_along_axis(nanos_to_millis, 0, result_times)

            #to_ts = np.vectorize(nanos_to_ts)
            timestamps_msec = np.apply_along_axis(lambda n: nanos_to_millis(n - timestamps_nanos[0]), 0, timestamps_nanos)

            err_timestamps_msec = None
            if test.errors is not None:
                err_nanos = test.errors[:, 0]
                err_timestamps_msec = np.apply_along_axis(lambda n: nanos_to_millis(n - err_nanos[0]), 0, err_nanos)

            hist = sorted(result_msec)
            #fit = stats.norm.pdf(hist, np.mean(hist), np.std(hist))

            html += "<h3>%s at %s</h3>" % (testname, hostname)
            plt.close()
            weights = np.ones_like(hist) / len(hist)
            plt.hist(hist,  histtype='stepfilled', normed=0, weights=weights, cumulative=True, color='b', label='Cumulative probability')
            plt.hist(hist,  histtype='stepfilled', normed=0, weights=weights, color='r', alpha=0.5, label='Probability')
            #plt.plot(hist, fit, color='y-o')
            plt.plot(hist, stats.norm.pdf(hist), 'y-', lw=2, label='pdf')
            plt.title("Normalized distribution of '%s' on %s" % (testname, hostname))
            plt.xlabel("Action time, msec")
            plt.ylabel("Probability")
            plt.legend()
            html += "<p>"
            html += save_figure(output_directory, "%s_%s_dist_normed.png", hostname, testname)

            plt.close()
            plt.hist(hist, bins=30, histtype='stepfilled', normed=False, color='b', label='Frequency')
            plt.title("Distribution of '%s' on %s" % (testname, hostname))
            plt.xlabel("Action time, msec")
            plt.ylabel("Frequency")
            plt.legend()
            html += save_figure(output_directory, "%s_%s_dist_freq.png", hostname, testname)
            html += "</p>"

            plt.close()
            plt.plot(timestamps_msec, result_msec, 'g-^')
            plt.gcf().autofmt_xdate()
            plt.title("History of '%s' on %s" % (testname, hostname))
            plt.xlabel("Time since test start, msec")
            plt.ylabel("Action time, msec")
            plt.legend()
            html += "<p>"
            html += save_figure(output_directory, "%s_%s_history.png", hostname, testname)

            if err_timestamps_msec is not None:
                plt.close()
                plt.hist(err_timestamps_msec, bins=err_timestamps_msec[-1] / 200, histtype='stepfilled', normed=False, color='r', label='Errors (total %d)' % len(err_timestamps_msec))
                plt.title("Error history of '%s' on %s" % (testname, hostname))
                plt.xlabel("Time since test start, msec")
                plt.ylabel("Errors count")
                plt.legend()
                html += save_figure(output_directory, "%s_%s_err_history.png", hostname, testname)

            html += "</p>"
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
    fnames = set(sys.argv[1:])
    output = tempfile.mkdtemp("stplotter")

    setup_logging()

    with open(os.path.join(output, "report.html"), "w+") as out:
        out.write("<html><body>")
        for fname in fnames:
            out.write(process_zip_report(fname, output))
        out.write("</body></html>")

    zipf = zipfile.ZipFile('report-%s.zip' % (datetime.now().strftime("%Y-%m-%d--%H-%M-%S")), 'w')
    zipdir(output, zipf)
    zipf.close()