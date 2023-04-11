#!/usr/bin/env python
import os
import contextlib
import pickle
import numpy as np

from ssrs import Simulator

try:
    from termcolor import colored
except ModuleNotFoundError:
    def colored(text, color=None, on_color=None, attrs=None):
        return text


def run_test(cfg,logfile='log.rtest'):
    """Run SSRS and compare against golden track data

    All tracks are required to be identical
    """
    testname = cfg.run_name

    print(testname,': Running...',end='',flush=True)
    with open(logfile, 'a') as f:
        with contextlib.redirect_stdout(f):
            with contextlib.redirect_stderr(f):
                sim = Simulator(cfg)
                sim.simulate_tracks()

    testdatapath = sim._get_tracks_fname(sim.case_ids[0], 0, sim.mode_data_dir) + '.pkl'
    fname = os.path.split(testdatapath)[1]
    golddatapath = os.path.join('gold_files',testname,fname)
    with open(testdatapath,'rb') as f:
        testdata = pickle.load(f)
    with open(golddatapath,'rb') as f:
        golddata = pickle.load(f)

    if len(testdata) != len(golddata):
        print(f'\r{testname} :',colored('FAIL','red'),
              '-- different number of tracks',
              f'(test {len(testdata)}, expected {len(golddata)})',
              '\033[K')
        return False

    Ntracks = len(testdata)
    passed = np.zeros(Ntracks)
    for itrack,testtrack in enumerate(testdata):
        goldtrack = golddata[itrack]
        if len(testtrack) == len(goldtrack):
            if np.all(testtrack == goldtrack):
                passed[itrack] = 1
    passrate = 100 * np.mean(passed)

    if passrate == 100:
        print(f'\r{testname} :',colored('PASS','green'),'\033[K')
        return True
    else:
        print(f'\r{testname} :',colored('FAIL','red'),
              f'-- {passrate:.1f}% of tracks match',
              '\033[K')
        return False

#==============================================================================
#==============================================================================
#==============================================================================
if __name__ == '__main__':

    from totw import case1, case2, case3, case4
    run_test(case1)
    run_test(case2)
    run_test(case3)
    run_test(case4)
    
