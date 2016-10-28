#!/usr/bin/env python

"""
This script will do pre-processing of input data
"""

from __future__ import print_function

import os
import argparse
from warnings import warn
import multiprocessing as mp
import numpy as np
import pandas as pd

from physics_selections import select_fatjets, is_baseline_event
from weights import get_rpv_params, get_bkg_xsec, get_sumw

class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).
    """
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])

def parse_args():
    """Parse the command line arguments"""
    parser = argparse.ArgumentParser('prepare_data')
    add_arg = parser.add_argument
    add_arg('input_file_list', nargs='+',
            help='Text file of input files')
    add_arg('-o', '--output-file', help='Output HDF5 file')
    add_arg('-n', '--max-events', type=int,
            help='Maximum number of events to read')
    add_arg('-p', '--num-workers', type=int, default=0,
            help='Number of concurrent worker processes')
    return parser.parse_args()

def xaod_to_numpy(files, max_events=None):
    """Converts the xAOD tree into numpy arrays with root_numpy"""
    import root_numpy as rnp

    # Branch name remapping for convenience
    branchMap = {
        'CaloCalTopoClustersAuxDyn.calEta' : 'clusEta',
        'CaloCalTopoClustersAuxDyn.calPhi' : 'clusPhi',
        'CaloCalTopoClustersAuxDyn.calE' : 'clusE',
        'AntiKt10LCTopoTrimmedPtFrac5SmallR20JetsAux.pt' : 'fatJetPt',
        'AntiKt10LCTopoTrimmedPtFrac5SmallR20JetsAux.eta' : 'fatJetEta',
        'AntiKt10LCTopoTrimmedPtFrac5SmallR20JetsAux.phi' : 'fatJetPhi',
        'AntiKt10LCTopoTrimmedPtFrac5SmallR20JetsAux.m' : 'fatJetM',
        'EventInfoAuxDyn.mcChannelNumber' : 'dsid',
        'EventInfoAuxDyn.mcEventWeights' : 'genWeight',
    }

    print('Now processing:', files)
    # Convert the files
    try:
        with suppress_stdout_stderr():
            entries = rnp.root2array(files, treename='CollectionTree',
                                     branches=branchMap.keys(), stop=max_events,
                                     warn_missing_tree=True)
    except IOError as e:
        warn('root2array gave an IOError: %s' % e)
        return None

    # Rename the branches
    entries.dtype.names = branchMap.values()
    return entries

def filter_xaod_to_numpy(files, max_events=None):
    """Processes some files by converting to numpy and applying filtering"""

    # Convert the data to numpy
    entries = xaod_to_numpy(files, max_events)
    if entries is None:
        return None

    # Get vectorized selection functions
    vec_select_fatjets = np.vectorize(select_fatjets, otypes=[np.ndarray])
    vec_select_baseline_events = np.vectorize(is_baseline_event)

    # Object selection
    selectedFatJets = vec_select_fatjets(entries['fatJetPt'], entries['fatJetEta'])
    # Baseline event selection
    baselineEvents = vec_select_baseline_events(entries['fatJetPt'], selectedFatJets)
    print('Baseline selected events: %d / %d' % (np.sum(baselineEvents), entries.size))

    return entries[baselineEvents]

def get_calo_image(entries, xkey='clusEta', ykey='clusPhi', wkey='clusE',
                   bins=100, xlim=[-2.5, 2.5], ylim=[-3.15, 3.15]):
    """Convert the numpy structure with calo clusters into 2D calo histograms"""
    def hist2d(x, y, w):
        return np.histogram2d(x, y, bins=bins, weights=w, range=[xlim, ylim])[0]
    return map(hist2d, entries[xkey], entries[ykey], entries[wkey])

def get_meta_data(entries):
    """Use the dsid to get sample metadata like xsec"""
    dsids = entries['dsid']
    # Try to get RPV metadata
    try:
        mglu, mneu, xsec = np.vectorize(get_rpv_params)(dsids)
    except KeyError:
        mglu, mneu, xsec = None, None, np.vectorize(get_bkg_xsec)(dsids)
    # Get the sum of generator weights
    sumw = np.vectorize(get_sumw)(dsids)
    return mglu, mneu, xsec, sumw

def get_event_weights(xsec, mcw, sumw, lumi=36000):
    """Calculate event weights"""
    return xsec * mcw * lumi / sumw

def main():
    """Main execution function"""
    args = parse_args()

    # Get the input file list
    input_files = []
    for input_list in args.input_file_list:
        with open(input_list) as f:
            input_files.extend(map(str.rstrip, f.readlines()))
    print('Processing %i input files' % len(input_files))
    #print(input_files)

    # Parallel processing
    if args.num_workers > 0:
        # Create a pool of workers
        pool = mp.Pool(processes=args.num_workers)
        # Convert to numpy structure in parallel
        entries = pool.map(filter_xaod_to_numpy, input_files)
        # Concatenate the results together
        entries = np.concatenate(filter(entries))
    # Sequential processing
    else:
        # Run the conversion and filter directly
        entries = filter_xaod_to_numpy(input_files, args.max_events)

    #print('Array shape and types:', entries.shape, entries.dtype)
    if entries.shape[0] == 0:
        print('No events selected by filter. Exiting.')
        return

    # Get the 2D histogram
    histos = get_calo_image(entries)

    # Get sample metadata
    mglu, mneu, xsec, sumw = get_meta_data(entries)

    # Calculate the event weights
    w = get_event_weights(xsec, entries['genWeight'], sumw)

    # Store results into pandas DF (??)
    df = pd.DataFrame.from_records(entries)
    df['clusHist'] = histos
    if mglu is not None: df['mGluino'] = mglu
    if mneu is not None: df['mNeutralino'] = mneu
    df['xsec'] = xsec
    df['sumw'] = sumw
    df['w'] = w
    print(df)

    # Write results to an HDF5 file
    if args.output_file is not None:
        # Save only the histos for now
        dfOut = df[['clusHist', 'w']]
        dfOut.to_hdf(args.output_file, 'main')

if __name__ == '__main__':
    main()
