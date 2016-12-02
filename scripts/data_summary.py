#!/usr/bin/env python

"""
This script can summarize the content of the HDF5 datasets
"""

from __future__ import print_function

import os
import argparse
import h5py
import numpy as np

def parse_args():
    """Parse the command line arguments"""
    parser = argparse.ArgumentParser('data_summary.py')
    add_arg = parser.add_argument
    add_arg('input_files', nargs='+', help='Input HDF5 files')
    return parser.parse_args()

def print_event(event):
    """
    Prints summary content of one event.
    Takes an HDF5 group corresponding to data for one event.
    """
    print(event.keys())
    print('Number of clusters:', event['clusE'].len())
    try:
        print('Gluino mass:', event['mGlu'].value)
        print('Neutralino mass:', event['mNeu'].value)
    except KeyError:
        pass
    print('Passes SR:', event['passSR'].value)
    print('Event weight:', event['weight'].value)

def get_flags(events):
    """Get numpy arrays of all the event weights and passSR"""
    weight = np.zeros((len(events)))
    passSR4J = np.zeros(len(events), bool)
    passSR5J = np.zeros(len(events), bool)
    passSR = np.zeros(len(events), bool)
    for i, event in enumerate(events):
        weight[i] = event['weight'].value
        passSR4J[i] = event['passSR4J'].value
        passSR5J[i] = event['passSR5J'].value
        passSR[i] = event['passSR'].value
    return weight, passSR4J, passSR5J, passSR

def main():
    args = parse_args()

    for input_file in args.input_files:
        print('Processing', os.path.basename(input_file))

        try:
            # Open the input file
            with h5py.File(input_file, 'r') as hf:
                events = hf.values()
                weight, passSR4J, passSR5J, passSR = get_flags(events)
                print('  Events in file:     ', len(events))
                print('  Events passing SR4J:', passSR4J.sum())
                print('  Events passing SR5J:', passSR5J.sum())
                print('  Events passing SR:  ', passSR.sum())
                print('  Weighted SR4J:      ', weight[passSR4J].sum())
                print('  Weighted SR5J:      ', weight[passSR5J].sum())
                print('  Weighted SR:        ', weight[passSR].sum())
                print('  First weight:       ', weight[0])

        except IOError as e:
            print('IOError:', e)

if __name__ == '__main__':
    main()
