"""
This module defines the physics code for the RPV multi-jet analysis with numpy.
That includes the jet object selection, baseline event selection, and signal
region event selection. It also provides the functionality to calculate the
summed jet mass physics variable.
"""

from __future__ import print_function
import numpy as np

class units():
    GeV = 1e3

class cuts():
    # Object selection
    fatjet_pt_min = 200*units.GeV
    fatjet_eta_max = 2.
    # Baseline event selection
    baseline_num_fatjet_min = 3
    baseline_fatjet_pt_min = 440*units.GeV
    # Signal region event selection
    sr_deta12_max = 1.4
    sr4j_mass_min = 800*units.GeV
    sr5j_mass_min = 600*units.GeV

def _apply_indices(a, indices):
    """Helper function for applying index array if it exists"""
    if indices is not None:
        return a[indices]
    else:
        return a

def filter_objects(obj_idx, *obj_arrays):
    """Applies an object filter to a set of object arrays."""
    filtered_arrays = []
    def filt(x, idx):
        return x[idx]
    vec_filter = np.vectorize(filt, otypes=[np.ndarray])
    for obj_array in obj_arrays:
        filtered_arrays.append(vec_filter(obj_array, obj_idx))
    return filtered_arrays

def filter_events(event_idx, *arrays):
    """Applies an event filter to a set of arrays."""
    return map(lambda x: x[event_idx], arrays)

def select_fatjets(fatjet_pts, fatjet_etas):
    """
    Selects the analysis fat jets for one event.

    Input params
      fatjet_pts: array of fat jet pt
      fatjet_etas: array of fat jet eta

    Returns a boolean index-array of the selected jets
    """
    return np.logical_and(
            fatjet_pts > cuts.fatjet_pt_min,
            np.fabs(fatjet_etas) < cuts.fatjet_eta_max)

def is_baseline_event(fatjet_pts, selected_fatjets=None):
    """
    Applies baseline event selection to one event.

    Inputs
      fatjet_pts: array of fat jet pt
      selected_fatjets: boolean index-array of selected fatjets in the array

    Returns a bool
    """
    pts = _apply_indices(fatjet_pts, selected_fatjets)
    # Fat-jet multiplicity requirement
    if pts.size < cuts.baseline_num_fatjet_min:
        return False
    # Fat-jet trigger plateau efficiency requirement
    if np.max(pts) < cuts.baseline_fatjet_pt_min:
        return False
    return True

def sum_fatjet_mass(fatjet_ms, selected_fatjets=None):
    """
    Calculates the summed fat jet mass.
    Uses the 4 leading selected fat jets.

    Inputs
      fatjet_ms: array of fat jet masses
      selected_fatjets: boolean index-array of selected fatjets in the array
    
    Returns a float
    """
    masses = _apply_indices(fatjet_ms, selected_fatjets)
    return np.sum(masses[:4])

def fatjet_deta12(fatjet_etas, selected_fatjets):
    """Delta-eta between leading fat-jets"""
    eta1, eta2 = _apply_indices(fatjet_etas, selected_fatjets)[:2]
    return abs(eta1 - eta2)

def is_signal_region_event(summed_mass, fatjet_pts, fatjet_etas,
                           selected_fatjets, is_baseline=None):
    """
    Applies signal region selection to one event.

    Inputs
      summed_mass: summed fatjet mass as calculated by sum_fatjet_mass function
      fatjet_pts: array of fat jet pt
      fatjet_etas: array of fat jet eta
      selected_fatjets: boolean index-array of selected fatjets in array
      is_baseline: whether the event passes baseline event selection

    Returns a bool
    """
    # Baseline event selection
    if is_baseline == False:
        return False
    if (is_baseline is None and
        not is_baseline_event(fatjet_pts, selected_fatjets)):
        return False
    # Fat-jet multiplicity
    num_fatjets = (np.sum(selected_fatjets)
                   if selected_fatjets is not None
                   else fatjet_etas.size)
    if num_fatjets < 4:
        return False
    # Delta-eta between leading fat-jets
    deta12 = fatjet_deta12(fatjet_etas, selected_fatjets)
    if deta12 > cuts.sr_deta12_max:
        return False
    # Summed jet mass cut
    elif num_fatjets == 4 and summed_mass < cuts.sr4j_mass_min:
        return False
    elif num_fatjets >= 5 and summed_mass < cuts.sr5j_mass_min:
        return False
    # Passes all requirements
    return True
