#!/bin/env python

"""
This script requires a full RootCore setup (e.g., via CVMFS on PDSF).
It is used to calculate the total number of simulated events in an xAOD
sample which can be used in weighting events to a desired luminosity.
"""

from __future__ import print_function

import sys
fileList = sys.argv[1]

print(fileList)

import ROOT
ROOT.gROOT.Macro('$ROOTCOREDIR/scripts/load_packages.C')
from ROOT import xAOD
if xAOD.Init().isFailure():
    raise Exception('xAOD Init failure')

sumw = 0

with open(fileList) as f:
    fileNames = map(str.rstrip, f.readlines())

for fileName in fileNames:

    # Construct transient meta tree
    f = ROOT.TFile(fileName)
    t = xAOD.MakeTransientMetaTree(f, 'MetaData')
    t.GetEntry(0)

    # Find the cut bookkeeper with largest cycle number
    maxCycle = -1
    for cbk in t.CutBookkeepers:
        if ( cbk.name() == 'AllExecutedEvents' and
             cbk.inputStream() == 'StreamAOD' and
             cbk.cycle() > maxCycle):
            maxCycle = cbk.cycle()
            theCbk = cbk

    # Accumulate
    sumw += theCbk.sumOfEventWeights()
    ROOT.xAOD.ClearTransientTrees()

print('Sum weights:', sumw)

