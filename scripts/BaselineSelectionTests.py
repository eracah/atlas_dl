from __future__ import print_function
import matplotlib; matplotlib.use("agg")



import functools
import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
import ROOT;
import sys
sys.path.append('/global/homes/w/wbhimji/cori-envs/nersc-rootpy/lib/python2.7/site-packages/')
import root_numpy as rnp


# # Loading the input into numpy
# 
# Using root_numpy to make structured arrays.


#filename = '/Users/sfarrell/Atlas/xaod/mc15_13TeV.361023.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ3W.merge.DAOD_EXOT3.e3668_s2576_s2132_r7728_r7676_p2613/DAOD_EXOT3.08204445._000002.pool.root.1'
filename = '/global/projecta/projectdirs/atlas/atlaslocalgroupdisk/rucio/mc15_13TeV/76/71/DAOD_EXOT3.08629754._000001.pool.root.1'



bg_files = [line.rstrip() for line in open('/global/project/projectdirs/das/wbhimji/RPVSusyJetLearn/atlas_dl/config/mc15_13TeV.361004.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ4.merge.DAOD_EXOT3.e3569_s2576_s2132_r7772_r7676_p2688-FileList.txt')]
sig_files = [line.rstrip() for line in open('/global/project/projectdirs/das/wbhimji/RPVSusyJetLearn/atlas_dl/config/mc15_13TeV.403568.MadGraphPythia8EvtGen_A14NNPDF23LO_GG_RPV10_1400_850.merge.DAOD_EXOT3.e5079_a766_a821_r7676_p2669-FileList.txt')]



# Branch names to read in and rename for convenience
branchMap = {
    'CaloCalTopoClustersAuxDyn.calEta' : 'ClusEta',
    'CaloCalTopoClustersAuxDyn.calPhi' : 'ClusPhi',
    'CaloCalTopoClustersAuxDyn.calE' : 'ClusE',
    'AntiKt10LCTopoTrimmedPtFrac5SmallR20JetsAux.pt' : 'FatJetPt',
    'AntiKt10LCTopoTrimmedPtFrac5SmallR20JetsAux.eta' : 'FatJetEta',
    'AntiKt10LCTopoTrimmedPtFrac5SmallR20JetsAux.phi' : 'FatJetPhi',
    'AntiKt10LCTopoTrimmedPtFrac5SmallR20JetsAux.m' : 'FatJetM',
}



entries = rnp.root2array(bg_files, treename='CollectionTree',
                         branches=branchMap.keys(),warn_missing_tree=True)
#                          start=0, stop=500000)
entries.dtype.names = branchMap.values()
print('Entries:', entries.size)



entries.dtype


# Since the data is structured, we can index by key name and do some fancy stuff.


# Multiple ways to dump variables for a specific event.
# I'm actually surprised these both work.
print(entries[0]['FatJetPt'])
print(entries['FatJetPt'][0])





# Perform object selections on one event
event = entries[3]
event['FatJetPt'] > 300000



# Select fatjets with pt > 200 GeV for all events in one go
f = np.vectorize(lambda jetPts: jetPts > 200000, otypes=[np.ndarray])
selectedJets = f(entries['FatJetPt'])
print(selectedJets)



# Select events with at least 2 selected jets
countSelected = np.vectorize(sum)
numJets = countSelected(selectedJets)
selectedEvents = numJets >= 2
print(numJets)
print(selectedEvents)


# ## Physics selections and variables
# Enough playing around. Let's test out the actual physics selections. The code has been put into the physics_selections module in the containing directory of this notebook.


sys.path.append('/project/projectdirs/das/wbhimji/RPVSusyJetLearn/atlas_dl_submitter/atlas_dl/scripts/')
from physics_selections import (select_fatjets, is_baseline_event,
                                sum_fatjet_mass, is_signal_region_event)



vec_select_fatjets = np.vectorize(select_fatjets, otypes=[np.ndarray])
vec_select_baseline_events = np.vectorize(is_baseline_event)
selectedFatJets = vec_select_fatjets(entries['FatJetPt'], entries['FatJetEta'])
baselineEvents = vec_select_baseline_events(entries['FatJetPt'], selectedFatJets)
print('Baseline selected events: %d / %d' % (np.sum(baselineEvents), entries.size))



# Calculate the summed jet mass for all events
summedMasses = np.vectorize(sum_fatjet_mass)(entries['FatJetM'], selectedFatJets)
print(summedMasses[baselineEvents])



vec_select_sr_events = np.vectorize(is_signal_region_event)
signalEvents = vec_select_sr_events(summedMasses, entries['FatJetPt'], entries['FatJetEta'],
                                    selectedFatJets, baselineEvents)
signalEntries = entries[signalEvents]
numSignalEvents = np.sum(signalEvents)
print('Signal events: %d / %d' % (numSignalEvents, entries.size))


# # Drawing signal region events


def get_hist2d(event):
    """Convert event into the calo-cluster image"""
    return np.histogram2d(event['ClusEta'], event['ClusPhi'],
                          bins=(50, 50), weights=event['ClusE'],
                          range=[[-2.5, 2.5], [-3.15, 3.15]])[0]

def plot_calo_image(h2d):
    """Plot a calo-image on the current axes"""
    plt.imshow(np.log10(h2d).T, #extent=[-2.,2.,-3.14, 3.14],
               extent=[-2.5, 2.5, -3.15, 3.15],
               interpolation='none', aspect='auto', origin='low')
    plt.colorbar(label='Cluster energy [Log(MeV)]')
    plt.xlabel('eta')
    plt.ylabel('phi')

def plot_jets(jetEtas, jetPhis, jetRadius=1):
    """Plot jet circles on the current axes"""
    for eta, phi in zip(jetEtas, jetPhis):
        circle = plt.Circle((eta, phi), radius=jetRadius, facecolor='none')
        plt.gcf().gca().add_artist(circle)



# Pick out a sample of signal region events.
# The indexing is now starting to get very confusing.
numSample = 4
sampleIdxs = np.random.choice(np.arange(numSignalEvents), numSample, replace=False)
sampleEntries = signalEntries[sampleIdxs]
sampleFatJets = selectedFatJets[signalEvents][sampleIdxs] # are we lost yet?
assert(sampleEntries.size == sampleFatJets.size)



# Get the quantities to plot
hists = [get_hist2d(ev) for ev in sampleEntries]
jetEtas = [etas[jets] for (etas, jets) in zip(sampleEntries['FatJetEta'], sampleFatJets)]
jetPhis = [phis[jets] for (phis, jets) in zip(sampleEntries['FatJetPhi'], sampleFatJets)]



# Draw the calo images and draw the selected fat jets as circles
plt.figure(figsize=(12, 10))
plt.subplot(221)
plot_calo_image(hists[0])
plot_jets(jetEtas[0], jetPhis[0])

plt.subplot(222)
plot_calo_image(hists[1])
plot_jets(jetEtas[1], jetPhis[1])

plt.subplot(223)
plot_calo_image(hists[2])
plot_jets(jetEtas[2], jetPhis[2])

plt.subplot(224)
plot_calo_image(hists[3])
plot_jets(jetEtas[3], jetPhis[3])

