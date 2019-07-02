#!/usr/bin/env python

from __future__ import print_function

import os
import argparse
import numpy as np
import h5py
import root_numpy as rnp

from utils import suppress_stdout_stderr

def get_calo_image(tree, xkey='clusEta', ykey='clusPhi', wkey='clusE',
                   bins=100, xlim=[-2.5, 2.5], ylim=[-3.15, 3.15]):
    """Convert the numpy structure with calo clusters into 2D calo histograms"""
    # Bin the data and reshape so we can concatenate along first axis into a 3D array.
    def hist2d(x, y, w):
        return (np.histogram2d(x, y, bins=bins, weights=w, range=[xlim, ylim])[0]
                .reshape([1, bins, bins]))
    hist_list = map(hist2d, tree[xkey], tree[ykey], tree[wkey])
    return np.concatenate(hist_list)

def get_tree(files, branch_dict, tree_name='CollectionTree', max_events=None):
    """Applies root_numpy to get out a numpy array"""
    # Convert the files
    try:
        with suppress_stdout_stderr():
            tree = rnp.root2array(files, treename=tree_name,
                                  branches=branch_dict.keys(), stop=max_events,
                                  warn_missing_tree=True)
    except IOError as e:
        print('WARNING: root2array gave an IOError:', e)
        return None

    # Rename the branches
    tree.dtype.names = branch_dict.values()
    return tree

def parse_args():
    """Parse the command line arguments"""
    parser = argparse.ArgumentParser('prepare_data')
    add_arg = parser.add_argument
    add_arg('input_file_list', nargs='+',
            help='Text file of input files')
    add_arg('-n', '--max-events', type=int,
            help='Maximum number of events to read')
    add_arg('--bins', default=64, type=int,
            help='The number of bins aka the dimensions of the hist data')
    add_arg('-o', '--output-np', help='Output compressed numpyfile')
    return parser.parse_args()


def main():    
    args = parse_args()
    input_files = []
    for input_list in args.input_file_list:
        with open(input_list) as f:
            input_files.extend(map(str.rstrip, f.readlines()))
    print('Processing %i %s input files' % (len(input_files), 'Delphes'))
    files=input_files
    if type(files) != list:
        files = [files]
    print('Now processing:', files)

    branch_dict = {
        'Tower.Eta' : 'clusEta',
        'Tower.Phi' : 'clusPhi',
        'Tower.E' : 'clusE',
        'Tower.Eem' : 'clusEM',
        'FatJet.PT' : 'fatJetPt',
        'FatJet.Eta' : 'fatJetEta',
        'FatJet.Phi' : 'fatJetPhi',
        'FatJet.Mass' : 'fatJetM',
        'Track.Eta' : 'trackEta',
        'Track.Phi' : 'trackPhi',
    }

    tree = get_tree(files, branch_dict, tree_name='Delphes')
    
    caloimages = get_calo_image(tree, bins=args.bins)

    np.save(args.output_np,caloimages)

if __name__ == '__main__':
    main()
