import pandas as pd
import string
import numpy as np

""" TO DO: implement shiftx2 parser."""

def read_sparta_tab(filename, skiprows):
    names = string.split("RESID RESNAME ATOMNAME SS_SHIFT SHIFT RC_SHIFT HM_SHIFT EF_SHIFT SIGMA")
    x = pd.io.parsers.read_table(filename, skiprows=skiprows, header=None, names=names, sep="\s*")
    x.rename(columns=lambda x: string.lower(x),inplace=True)  # Make lowercase names
    x.rename(columns={"atomname":"name"}, inplace=True)  # let atomname be called name.
    x["experiment"] = "CS"
    x["name"] = x["name"].map(lambda x: x.replace("HN","H"))
    x = x.pivot_table(rows=["experiment", "resid", "name"], values=["shift"])
    return x

    
def read_all_sparta(filenames, skiprows):
    num_frames = len(filenames)

    filename = filenames[0]
    x = read_sparta_tab(filename, skiprows)
    num_measurements = x.shape[0]
    
    d = pd.DataFrame(np.zeros((num_frames, num_measurements)), columns=x.index)
    for k, filename in enumerate(filenames):
        x = read_sparta_tab(filename, skiprows)
        d.iloc[k] = x["shift"]

    return d
    
def read_ppm_data(filename):
    x = pd.io.parsers.read_table(filename, header=None, sep="\s*")
    res_id = x.iloc[:,0]
    res_name = x.iloc[:,1]
    atom_name = x.iloc[:,2]
    values = x.iloc[:,4:].values
    #indices = ["CS_%d_%s" % (res_id[i], atom_name[i]) for i in range(len(res_id))]
    expt = ["CS" for r in res_id]
    indices = pd.MultiIndex.from_arrays((expt, res_id, atom_name), names=("experiment", "resid", "name"))
    d = pd.DataFrame(values.T, columns=indices)
    return d

def read_shiftx2_intermediate(directory):
    atom_name = np.loadtxt(directory + "/shifts_atoms.txt", "str")
    res_id = np.loadtxt(directory + "/shifts_resid.dat", 'int')
    values = np.load(directory + "/shifts.npz")["arr_0"]
    #indices = ["CS_%d_%s" % (res_id[i], atom_name[i]) for i in range(len(res_id))]
    expt = ["CS" for r in res_id]
    indices = pd.MultiIndex.from_arrays((expt, res_id, atom_name), names=("experiment", "resid", "name"))
    d = pd.DataFrame(values, columns=indices)
    return d

def read_shiftx2(filename):
    x = pd.io.parsers.read_csv(filename)  # NUM,RES,ATOMNAME,SHIFT
    x.rename(columns=lambda x: string.lower(x),inplace=True)  # Make lowercase names
    x.rename(columns={"num":"resid", "atomname":"name"}, inplace=True)  # let atomname be called name.
    x["experiment"] = "CS"
    x = x.pivot_table(rows=["experiment", "resid", "name"])
    return x

def read_all_shiftx2(filenames):
    num_frames = len(filenames)

    filename = filenames[0]
    x = read_shiftx2(filename)
    num_measurements = x.shape[0]
    
    d = pd.DataFrame(np.zeros((num_frames, num_measurements)), columns=x.index)
    for k, filename in enumerate(filenames):
        x = read_shiftx2(filename)
        d.iloc[k] = x["shift"]

    return d
