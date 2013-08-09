import pandas as pd
import numpy as np

ppm_atom_uncertainties = pd.Series({
"CA":1.06,
"CB":1.23,
"C":1.32,
"H":0.53,
"N":2.91
})

shiftx2_atom_uncertainties = pd.Series({  # This uses shiftx+ RMS, which only accounts for geometric effects.  More realistic for comparing structures of the same sequence.
"N":2.0862,
"CA":0.7743,
"CB":0.8583,
"C": 0.8699,
"H": 0.3783,
"HA":0.1967,
"HA2":0.1967,
"HA3":0.1967,
})

shiftx2_atom_uncertainties_BAD = pd.Series({  # This uses overall shiftx2 RMS, which includes the sequence effects.  We don't care about sequence differences, so these are overly optimistic.
"N":1.1169,
"CA":0.4412,
"CB":0.5163,
"C": 0.5330,
"H": 0.1711,
"HA":0.1231,
})

sparta_atom_uncertainties = pd.Series({
"N":2.45,
"C":1.07,
"CA":0.92,
"CB":1.13,
"HA":0.25,
"HA2":0.25,
"HA3":0.25,
"H":0.49
})

all_atom_uncertainties = pd.DataFrame([sparta_atom_uncertainties, shiftx2_atom_uncertainties, ppm_atom_uncertainties], index=["sparta","shiftx2","ppm"])
weights = all_atom_uncertainties ** -2.
weights /= weights.sum()
weights["HA"]["ppm"] = 0.
weights["HA2"]["ppm"] = 0.
weights["HA3"]["ppm"] = 0.

atom_uncertainties_independent = ((weights * all_atom_uncertainties) ** 2.0).sum() ** 0.5
atom_uncertainties = (weights * all_atom_uncertainties ** 2.0).sum() ** 0.5


old_cs_atom_uncertainties = pd.Series({
#"N":2.0862,"CA":0.7743,"CB":0.8583,"C":0.8699,"H":0.3783,"HA":0.1967         # ShiftX+ V1.07
"CS_2_N":2.4625,"CS_2_CA":0.7781,"CS_2_CB":1.1760,"CS_2_C":1.1309,"CS_2_H":0.4685,"CS_2_HA":0.2743,  # Mean uncertainties from SPARTA+
})


def reweight(dataframe, prediction_model):
    atom_scales = weights.ix[prediction_model]
    atoms = dataframe.columns.get_level_values("name")
    sig = weights.ix[prediction_model]
    scale = pd.Series(sig[atoms].values, index=dataframe.columns)
    dataframe = dataframe * scale
    return dataframe
