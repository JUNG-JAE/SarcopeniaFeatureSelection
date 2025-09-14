
# Basic features are used to approximate ASM value

# Obesity(3)
Obesity = ['BFM_kg', 'Percent_BF', 'BMR_kcal']

# Blood pressure(4) 
BP = ['SBP_mmHg', 'DBP_mmHg', 'BP_Stage', 'Pulse'] 

# Plartar, Dorsal strength(4) 
PD = ['D_Plantar', 'D_Dorsal', 'ND_Plantar', 'ND_Dorsal']

# Single leg stance test(2)
SLS = ['D_SLS', 'ND_SLS']

# Physical Test(4)
PT = ['SS', 'CSR', 'X2MWT', 'TUG']

# Grade(5)
G = ['G_SS', 'G_CSR', 'G_2MWT', 'G_TUG', 'G_D_SLS']

# Survey(9)
Svy = ['D1_s', 'D2_s', 'D3_s', 'D4_s', 'D5_s', 'D6_s', 'D7_s', 'SarQoL_Total_s', 'FES']

# Disease(4)
Dss = ['DM', 'Hypertension', 'Hyperlipidemia', 'Sleepdisorder']

# etc(2)
so = ['HbA1c', 'SAF']

# for extra dataset(5)
Resp = ['FVC', 'PreFVC', 'FEV1', 'PEF', 'MIP_Ave']

# SES = ['Educationlevel', 'Educationlevel_p', 'Income', 'Smoking_', 'Smoking_d_', 'Smoking_a_', 'Drinking_f_', 'Drinking_d_', 'Drinking_a_', 'RegularPA_', 'TypeofPA', 'Religion', 'House', 'Family']
# SES = ['Smoking_', 'Smoking_d_', 'RegularPA_', 'Religion', 'House', 'Family']
SES = ['Smoking_', 'Smoking_d_', 'Drinking_f_', 'RegularPA_', 'Religion', 'House', 'Family']

all_features = Obesity + BP + PD + SLS + PT + G + Svy + Dss + so + SES # + Resp

# total (56) - resp (5) = final (51)

# scaler should be used at continous features not discrete features
# binary_nominal(6)
bin_features = ['DM', 'Hypertension', 'Hyperlipidemia', 'Sleepdisorder', 'Smoking_',  'RegularPA_']

# categorical_ordinal(14)
cate_ordinal_features = ['BP_Stage', 'G_SS', 'G_CSR', 'G_2MWT', 'G_TUG', 'G_D_SLS', 'Educationlevel',  'Educationlevel_p', 
                         'Income', 'Smoking_d_', 'Smoking_a_', 'Drinking_f_', 'Drinking_d_', 'Drinking_a_' ]

# categorical_nominal(4) which should have to be one-hot encoded
# cate_nomial_features = ['Religion', 'House', 'Family', 'TypeofPA']
cate_nomial_features = ['Religion', 'House', 'Family']

non_cont_features = bin_features + cate_ordinal_features + cate_nomial_features

# continuous(32)
cont_features = [feature for feature in all_features if feature not in bin_features + cate_ordinal_features + cate_nomial_features]
