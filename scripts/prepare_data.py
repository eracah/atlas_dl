#!/usr/bin/env python

"""
This script will do pre-processing of input data
"""

from __future__ import print_function

import os
import argparse
import multiprocessing as mp
import numpy as np
import h5py
import root_numpy as rnp

from physics_selections import (filter_objects, filter_events,
                                select_fatjets, is_baseline_event,
                                sum_fatjet_mass, fatjet_deta12,
                                pass_sr4j, pass_sr5j,
                                is_signal_region_event)
from weights import (get_xaod_rpv_params, get_xaod_bkg_xsec, get_xaod_sumw,
                     get_delphes_xsec, get_delphes_sumw)
from utils import suppress_stdout_stderr


def parse_args():
    """Parse the command line arguments"""
    parser = argparse.ArgumentParser('prepare_data')
    add_arg = parser.add_argument
    add_arg('input_file_list', nargs='+',
            help='Text file of input files')
    add_arg('--input-type', default='xaod', choices=['xaod', 'delphes'],
            help='Specify xaod or delphes input file type')
    add_arg('-o', '--output-npz', help='Output compressed numpy binary file')
    add_arg('--output-h5', help='Output hdf5 file')
    add_arg('-n', '--max-events', type=int,
            help='Maximum number of events to read')
    add_arg('-p', '--num-workers', type=int, default=0,
            help='Number of concurrent worker processes')
    add_arg('--write-clus', action='store_true',
            help='Write cluster info to output')
    add_arg('--write-fjets', action='store_true',
            help='Write fat jet info to output')
    add_arg('--write-mass', action='store_true',
            help='Write RPV theory mass params to output')
    add_arg('--write-tracks', action='store_true',
            help='Write ID track variables')
    add_arg('--bins', default=64, type=int,
            help='The number of bins aka the dimensions of the hist data')
    return parser.parse_args()

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

def process_events(tree):
    """Applies physics selections and filtering"""

    # Object selection
    vec_select_fatjets = np.vectorize(select_fatjets, otypes=[np.ndarray])
    fatJetPt, fatJetEta = tree['fatJetPt'], tree['fatJetEta']
    jetIdx = vec_select_fatjets(fatJetPt, fatJetEta)

    fatJetPt, fatJetEta, fatJetPhi, fatJetM = filter_objects(
        jetIdx, fatJetPt, fatJetEta, tree['fatJetPhi'], tree['fatJetM'])

    # Baseline event selection
    skimIdx = np.vectorize(is_baseline_event)(fatJetPt)
    fatJetPt, fatJetEta, fatJetPhi, fatJetM = filter_events(
        skimIdx, fatJetPt, fatJetEta, fatJetPhi, fatJetM)
    num_baseline = np.sum(skimIdx)
    print('Baseline selected events: %d / %d' % (num_baseline, tree.size))

    # Calculate quantities needed for SR selection
    if num_baseline > 0:
        numFatJet = np.vectorize(lambda x: x.size)(fatJetPt)
        sumFatJetM = np.vectorize(sum_fatjet_mass)(fatJetM)
        fatJetDEta12 = np.vectorize(fatjet_deta12)(fatJetEta)

        # Signal-region event selection
        passSR4J = np.vectorize(pass_sr4j)(numFatJet, sumFatJetM, fatJetDEta12)
        passSR5J = np.vectorize(pass_sr5j)(numFatJet, sumFatJetM, fatJetDEta12)
        passSR = np.logical_or(passSR4J, passSR5J)
    else:
        numFatJet = sumFatJetM = fatJetDEta12 = np.zeros(0)
        passSR4J = passSR5J = passSR = np.zeros(0, dtype=np.bool)

    # Return results in a dict of arrays
    return dict(tree=tree[skimIdx], fatJetPt=fatJetPt, fatJetEta=fatJetEta,
                fatJetPhi=fatJetPhi, fatJetM=fatJetM, sumFatJetM=sumFatJetM,
                passSR4J=passSR4J, passSR5J=passSR5J, passSR=passSR)

def filter_delphes_to_numpy(files, max_events=None):
    """Processes some files by converting to numpy and applying filtering"""

    if type(files) != list:
        files = [files]

    # Branch name remapping for convenience
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

    # Convert the data to numpy
    print('Now processing:', files)
    tree = get_tree(files, branch_dict, tree_name='Delphes', max_events=max_events)
    if tree is None:
        return None
    # Fix units
    for key in ['clusE', 'clusEM', 'fatJetPt', 'fatJetM']:
        tree[key] = tree[key]*1e3

    # Apply physics
    results = process_events(tree)
    skimTree = results['tree']

    # Move the track kinematics for consistency with xaod
    for key in ['trackEta', 'trackPhi']:
        results[key] = skimTree[key]

    # Get the sample config string from the filenames.
    # For now, out of laziness, allow only one sample at a time.
    # NOTE: this assumes a particular naming convention of the delphes files!!
    samples = map(lambda s: os.path.basename(s).split('-')[0], files)
    if np.unique(samples).size > 1:
        raise Exception('Mixing delphes samples not yet supported: ' + str(samples))

    # Store the sample name for metadata lookups
    num_event = results['tree'].shape[0]
    results['sample'] = np.full(num_event, samples[0], 'S30')

    return results

def filter_xaod_to_numpy(files, max_events=None):
    """Processes some files by converting to numpy and applying filtering"""
    # Branch name remapping for convenience
    branch_dict = {
        'CaloCalTopoClustersAuxDyn.calEta' : 'clusEta',
        'CaloCalTopoClustersAuxDyn.calPhi' : 'clusPhi',
        'CaloCalTopoClustersAuxDyn.calE' : 'clusE',
        'CaloCalTopoClustersAuxDyn.EM_PROBABILITY' : 'clusEM',
        'AntiKt10LCTopoTrimmedPtFrac5SmallR20JetsAux.pt' : 'fatJetPt',
        'AntiKt10LCTopoTrimmedPtFrac5SmallR20JetsAux.eta' : 'fatJetEta',
        'AntiKt10LCTopoTrimmedPtFrac5SmallR20JetsAux.phi' : 'fatJetPhi',
        'AntiKt10LCTopoTrimmedPtFrac5SmallR20JetsAux.m' : 'fatJetM',
        'EventInfoAuxDyn.mcChannelNumber' : 'dsid',
        'EventInfoAuxDyn.mcEventWeights' : 'genWeight',
        'InDetTrackParticlesAuxDyn.theta' : 'trackTheta',
        'InDetTrackParticlesAuxDyn.phi' : 'trackPhi',
    }
    # Convert the data to numpy
    print('Now processing:', files)
    tree = get_tree(files, branch_dict, tree_name='CollectionTree',
                    max_events=max_events)
    if tree is None:
        return None
    # Apply physics
    results = process_events(tree)
    skimTree = results['tree']

    # Get the track coordinates
    vtan = np.vectorize(np.tan, otypes=[np.ndarray])
    vlog = np.vectorize(np.log, otypes=[np.ndarray])
    trackTheta = skimTree['trackTheta']
    results['trackEta'] = -vlog(vtan(trackTheta / 2))
    results['trackPhi'] = skimTree['trackPhi']

    return results

def get_calo_image(tree, xkey='clusEta', ykey='clusPhi', wkey='clusE',
                   bins=100, xlim=[-2.5, 2.5], ylim=[-3.15, 3.15]):
    """Convert the numpy structure with calo clusters into 2D calo histograms"""
    # Bin the data and reshape so we can concatenate along first axis into a 3D array.
    def hist2d(x, y, w):
        return (np.histogram2d(x, y, bins=bins, weights=w, range=[xlim, ylim])[0]
                .reshape([1, bins, bins]))
    hist_list = map(hist2d, tree[xkey], tree[ykey], tree[wkey])
    return np.concatenate(hist_list)

def get_track_image(tree, xkey='trackEta', ykey='trackPhi',
                   bins=100, xlim=[-2.5, 2.5], ylim=[-3.15, 3.15]):
    """Convert the numpy structure with calo clusters into 2D calo histograms"""
    # Bin the data and reshape so we can concatenate along first axis into a 3D array.
    def hist2d(x, y):
        return (np.histogram2d(x, y, bins=bins, range=[xlim, ylim])[0]
                .reshape([1, bins, bins]))
    hist_list = map(hist2d, tree[xkey], tree[ykey])
    return np.concatenate(hist_list)

def merge_results(dicts):
    """Merge a list of dictionaries with numpy arrays"""
    dicts = filter(None, dicts)
    # First, get the list of unique keys
    keys = set(key for d in dicts for key in d.keys())
    result = dict()
    for key in keys:
        arrays = [d[key] for d in dicts]
        result[key] = np.concatenate([d[key] for d in dicts])
    return result

def get_meta_data_delphes(sample_names):
    if sample_names is None:
        print('WARNING: no sample_names => no metadata => no event weights')
        return None, None, None, None
    # TODO: parse these out from the sample name
    mglu, mneu = None, None
    xsec = np.vectorize(get_delphes_xsec)(sample_names)
    sumw = np.vectorize(get_delphes_sumw)(sample_names)
    return mglu, mneu, xsec, sumw

def get_meta_data_xaod(dsids):
    """Use the dsid to get sample metadata like xsec"""
    if dsids is None:
        print('WARNING: no dsid => no metadata => no event weights')
        return None, None, None, None
    # Try to get RPV metadata
    try:
        mglu, mneu, xsec = np.vectorize(get_xaod_rpv_params)(dsids)
    except KeyError:
        mglu, mneu, xsec = None, None, np.vectorize(get_xaod_bkg_xsec)(dsids)
    # Get the sum of generator weights
    sumw = np.vectorize(get_xaod_sumw)(dsids)
    return mglu, mneu, xsec, sumw

def get_event_weights(xsec, mcw, sumw, lumi=36000):
    """Calculate event weights"""
    # Need to extract the first entry of the generator weights per event
    if type(mcw) == np.ndarray:
        mcw = np.vectorize(lambda g: g[0])(mcw)
    return xsec * mcw * lumi / sumw

def write_hdf5(filename, outputs):
    """
    Write the output dictionary contents to an hdf5 file.
    This will write one dataset group per event.
    """
    # Check that event count is consistent across all arrays
    lengths = np.array([a.shape[0] for a in outputs.values()])
    assert(np.all(lengths == lengths[0]))

    # Open the output file
    with h5py.File(filename, 'w') as hf:
        # Create one big h5f group
        g = hf.create_group('all_events')
        for key, data in outputs.iteritems():
            if key in ["hist", "histEM", "histtrack", "passSR4J", "passSR5J", "passSR", "weight"]:
                g.create_dataset(key, data=data)
        # Loop over events to write
        num_entries = outputs.values()[0].shape[0]
        for i in xrange(num_entries):
            # Create a group for this event
            g = hf.create_group('event_{}'.format(i))
            # Add the data for this event
            for key, data in outputs.iteritems():
                g.create_dataset(key, data=data[i])

def main():
    """Main execution function"""
    args = parse_args()

    # Get the input file list
    input_files = []
    for input_list in args.input_file_list:
        with open(input_list) as f:
            input_files.extend(map(str.rstrip, f.readlines()))
    print('Processing %i %s input files' % (len(input_files), args.input_type))

    # Configure for delphes or xaod
    if args.input_type == 'xaod':
        filter_func = filter_xaod_to_numpy
    else:
        filter_func = filter_delphes_to_numpy

    # Parallel processing
    if args.num_workers > 0:
        print('Starting process pool of %d workers' % args.num_workers)
        # Create a pool of workers
        pool = mp.Pool(processes=args.num_workers)
        # Convert to numpy structure in parallel
        task_data = pool.map(filter_func, input_files)
        # Merge the results from each task
        data = merge_results(task_data)
    # Sequential processing
    else:
        # Run the conversion and filter directly
        data = filter_func(input_files, args.max_events)

    tree = data['tree']
    if tree.shape[0] == 0:
        print('No events selected by filter. Exiting.')
        return

    # TODO: put all data in the data dict

    # Get the 2D histogram
    data['hist'] = get_calo_image(tree, bins=args.bins)
    data['histEM'] = get_calo_image(tree, xkey='clusEta', ykey='clusPhi', wkey='clusEM', bins=args.bins)
    data['histtrack'] = get_track_image(tree, bins=args.bins)

    # Get sample metadata
    if args.input_type == 'xaod':
        mglu, mneu, xsec, sumw = get_meta_data_xaod(tree['dsid'])
        mcw = tree['genWeight']
    else:
        mglu, mneu, xsec, sumw = get_meta_data_delphes(data.get('sample', None))
        mcw = 1

    # Calculate the event weights
    data['weight'] = (get_event_weights(xsec, mcw, sumw)
                      if sumw is not None else None)

    # Signal region flags
    passSR4J = data['passSR4J']
    passSR5J = data['passSR5J']
    passSR = data['passSR']

    # Dictionary of output data
    outputs = {}
    output_keys = ['hist', 'histEM', 'histtrack', 'passSR4J', 'passSR5J', 'passSR', 'weight']

    # Addition optional outputs
    if args.write_clus:
        for key in ['clusEta', 'clusPhi', 'clusE', 'clusEM']:
            try:
                outputs[key] = tree[key]
            except KeyError:
                print('Failed to write missing key:', key)
    if args.write_fjets:
        output_keys += ['fatJetPt', 'fatJetEta', 'fatJetPhi', 'fatJetM']
    if args.write_mass:
        if mglu is not None:
            outputs['mGlu'] = mglu
        if mneu is not None:
            outputs['mNeu'] = mneu
    if args.write_tracks:
        output_keys += ['trackEta', 'trackPhi']

    for key in output_keys:
        try:
            outputs[key] = data[key]
        except KeyError:
            print('Failed to write missing key:', key)
            raise

    # Print some summary information
    print('SR4J selected events: %d / %d' % (np.sum(passSR4J), tree.size))
    weight = data['weight']
    if weight is not None:
        print('SR4J weighted events: %f' % np.sum(weight[passSR4J]))
    print('SR5J selected events: %d / %d' % (np.sum(passSR5J), tree.size))
    if weight is not None:
        print('SR5J weighted events: %f' % np.sum(weight[passSR5J]))
    print('SR selected events: %d / %d' % (np.sum(passSR), tree.size))
    if weight is not None:
        print('SR weighted events: %f' % (np.sum(weight[passSR])))

    # Write results to compressed npz file
    if args.output_npz is not None:
        print('Writing output to', args.output_npz)
        np.savez_compressed(args.output_npz, **outputs)

    # Write results to hdf5
    if args.output_h5 is not None:
        print('Writing output to', args.output_h5)
        write_hdf5(args.output_h5, outputs)

    # TODO: Add support to write out a ROOT file..?

    print('Done!')

if __name__ == '__main__':
    main()
