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
from jinja2 import *

logger = logging.getLogger('stplot')
import matplotlib
matplotlib.rcParams.update({'font.size': 9})
FORMAT = "png"

def nanos_to_millis(nanos):
    return nanos * 10 ** -6

def nanos_to_ts(nanos):
    return datetime.fromtimestamp(int(nanos) // 1000000000)

def save_figure(fig, output_directory, fname):
    targetsubdir = "graphs"
    targetdir = os.path.join(output_directory, targetsubdir)

    try:
        os.makedirs(targetdir)
    except:
        pass

    name = os.path.join(targetdir, fname)
    fig.savefig(name, dpi=100)
    return os.path.join(targetsubdir, fname)

def zipdir(path, zip):
    for root, dirs, files in os.walk(path):
        for file in files:
            zip.write(os.path.join(root, file), arcname=os.path.join(os.path.relpath(root, path), file))


class Test(object):
    def __init__(self, hostname, testname, success_history, errors):
        self.hostname = hostname
        self.testname = testname
        self.success_history = success_history
        self.errors = errors
        
class TestSuiteRepr(object):
    def __init__(self, zipname):
        self.zipname = zipname
        self.tests = []

    def latest(self):
        return self.tests[-1]
        
        
class TestRepr(object):
    def __init__(self, name, hostname):
        self.name = name
        self.hostname = hostname
        self.dist = None
        self.freq = None
        self.errors = None
        self.hist = None


def process_test(output_directory, suiteRepr, test, metrics):
    hostname = test.hostname
    testname = test.testname
    history = test.success_history
    timestamps_nanos = history[:, 0]
    result_times = history[:, 1]
    result_msec = np.apply_along_axis(nanos_to_millis, 0, result_times)
    # to_ts = np.vectorize(nanos_to_ts)
    timestamps_msec_orig = np.apply_along_axis(nanos_to_millis, 0, timestamps_nanos)
    timestamps_msec = np.apply_along_axis(lambda n: n - timestamps_msec_orig[0], 0, timestamps_msec_orig)
    err_timestamps_msec = None
    if test.errors is not None:
        err_nanos = test.errors[:, 0]
        err_timestamps_msec = np.apply_along_axis(lambda n: nanos_to_millis(n - err_nanos[0]), 0, err_nanos)
    hist = sorted(result_msec)
    # fit = stats.norm.pdf(hist, np.mean(hist), np.std(hist))
    suiteRepr.tests.append(TestRepr(testname, hostname))
    plt.close()

    weights = np.ones_like(hist) / len(hist)
    plt.hist(hist, histtype='bar', normed=0, weights=weights, cumulative=True, color="#6495ed",
             label='Cumulative probability')
    plt.hist(hist, histtype='step', normed=0, weights=weights, color='#00008b', lw=4, label='Probability')
    # plt.plot(hist, fit, color='y-o')
    plt.plot(hist, stats.norm.pdf(hist), '-', color="#b0e0e6", lw=4, label='pdf')
    plt.title("Normalized distribution of '%s' on %s" % (testname, hostname))
    plt.xlabel("Action time, msec")
    plt.ylabel("Probability")
    plt.legend()
    suiteRepr.latest().dist = save_figure(plt, output_directory, "%s_%s_dist_normed.%s" % (hostname, testname, FORMAT))

    plt.close()
    plt.hist(hist, bins=30, histtype='bar', normed=False, color="#6495ed", label='Frequency')
    plt.title("Distribution of '%s' on %s" % (testname, hostname))
    plt.xlabel("Action time, msec")
    plt.ylabel("Frequency")
    plt.legend()
    suiteRepr.latest().freq = save_figure(plt, output_directory, "%s_%s_dist_freq.%s" % (hostname, testname, FORMAT))

    plt.close()
    #plt.axis('off')

    fig = plt.figure(facecolor='white')
    #fig.set_title("History of '%s' on %s" % (testname, hostname))
    left, width = 0.1, 0.8
    rect1 = [left, 0.7, width, 0.2]
    rect2 = [left, 0.3, width, 0.4]
    rect3 = [left, 0.1, width, 0.2]

    ax1 = fig.add_axes(rect1)  # left, bottom, width, height
    ax3 = fig.add_axes(rect2, sharex=ax1)
    ax4 = fig.add_axes(rect3, sharex=ax1)

    axhist = ax3
    if err_timestamps_msec is not None:
        axhist.hist(err_timestamps_msec, bins=err_timestamps_msec[-1] / 250, histtype='barstacked',
                 color='#fa8072', label='Errors (total %d)' % len(err_timestamps_msec))
        axhist.set_xlabel("Time since test start, msec")
        axhist.set_ylabel("Errors count")
        axhist.legend()
        axhist.set_yticks(axhist.get_yticks()[1:-1])
        axhist = axhist.twinx()
    axhist.plot(timestamps_msec, result_msec, '-o')
    axhist.set_xlabel("Time since test start, msec")
    axhist.set_ylabel("Action time, msec")

    axhist.set_yticks(axhist.get_yticks()[1:-1])

    axtime = ax1
    axtime.set_yticks([])
    axtime.set_xlim([0, timestamps_msec[-1] - timestamps_msec[0]])
    metricsts = np.copy(metrics.timestamps) - timestamps_msec_orig[0]
    axtime.plot(metricsts, metrics.proc_cpu_time, 'b-', label="process CPU time")
    axtime.legend(loc='upper left', framealpha=0.5)
    axtime.set_yticks(axtime.get_yticks()[1:-1])
    axtime = axtime.twinx()
    axtime.set_xlim([0, timestamps_msec[-1] - timestamps_msec[0]])
    axtime.plot(metricsts, metrics.sys_cpu_time, 'r-', label="system CPU time")
    axtime.legend(loc='upper right',framealpha=0.5)
    axtime.set_yticks(axtime.get_yticks()[1:-1])

    axload = ax4
    axload.set_xlim(axtime.get_xlim())
    axload.plot(metricsts, metrics.proc_cpu, 'b-', label="proc CPU%")
    axload.legend(loc='upper left', framealpha=0.5)
    axload.set_yticks(axload.get_yticks()[1:-1])
    axload = ax4.twinx()
    axload.set_xlim(axtime.get_xlim())
    axload.plot(metricsts, metrics.sys_la, 'r-', label="system la")
    axload.legend(loc='upper right', framealpha=0.5)
    axload.set_yticks(axload.get_yticks()[1:-1])


    suiteRepr.latest().hist = save_figure(fig, output_directory, "%s_%s_history.%s" % (hostname, testname, FORMAT))


class Metrics(object):
    def __init__(self, rawmetrics):
        timestamps_nanos = rawmetrics[:, 0]
        self.timestamps = np.apply_along_axis(lambda n: nanos_to_millis(n), 0, timestamps_nanos)
        self.proc_cpu = rawmetrics[:, 2]
        self.proc_cpu_time = np.apply_along_axis(lambda n: nanos_to_millis(n) // 10**3, 0, rawmetrics[:, 4])
        self.sys_cpu_time = rawmetrics[:, 6]
        self.sys_la = rawmetrics[:, 8]

def process_zip_report(fname, output_directory):
    zip = zipfile.ZipFile(fname)

    names = zip.namelist()
    zipname = os.path.split(fname)[1]
    success_history = []

    metrics = {}

    for n in names:
        parts = n.split(os.sep)
        logger.info("Processing %s::%s" % (zipname, n))

        if parts[1] == "calls_successful":
            hostname = parts[0]
            testname = os.path.splitext(parts[2])[0]

            if not hostname in metrics:
                with zip.open("%s/test_metrics.txt" % hostname, 'r') as content:
                    result = np.genfromtxt(content, delimiter=";")
                    metrics[hostname] = Metrics(result)

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

    suiteRepr = TestSuiteRepr(zipname)
    for test in success_history:
        process_test(output_directory, suiteRepr, test, metrics[test.hostname])

    return suiteRepr


def setup_logging():
    logger = logging.getLogger('stplot')
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def main(fnames):
    if True:
        output = "stplotterout"
        try:
            shutil.rmtree(output)
        except Exception:
            pass
        try:
            os.makedirs(output)
        except Exception:
            pass
    else:
        output = tempfile.mkdtemp("stplotter")

    setup_logging()
    env = Environment(loader=PackageLoader('stemplates', '.'))
    template = env.get_template('report.html')
    suites = []

    for fname in fnames:
        logger.info("Working on %s" % fname)
        suites.append(process_zip_report(fname, output))

    with open(os.path.join(output, "report.html"), "w+") as out:
        out.write(template.render(suites=suites))

    zipf = zipfile.ZipFile('report-%s.zip' % (datetime.now().strftime("%Y-%m-%d--%H-%M-%S")), 'w')
    zipdir(output, zipf)
    zipf.close()


if __name__ == '__main__':
    fnames = set(sys.argv[1:])
    main(fnames)