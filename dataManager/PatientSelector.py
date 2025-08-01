# CancerPainClassifier
# Copyright (c) 2025 Neeko
# License: MIT
# If used in research, please cite: https://github.com/Neeko-strong-tomato/CancerPainClassifier

import pandas as pd
import os

class PatientSelector:

    # "~/Desktop/Cancer_pain_data/PETdata/patients list/patient_list.xlsx", 
    #  "~/Desktop/Cancer_pain_data/PETdata/patients list/deID_CC_HN220303.xlsx"
    def __init__(self, file1_path = os.path.expanduser("~/Documents/CancerPain/PETdata/patients list/patient_list.xlsx"), 
                 file2_path = os.path.expanduser("~/Documents/CancerPain/PETdata/patients list/deID_CC_HN220303.xlsx") ):
        self.df1 = pd.read_excel(file1_path)
        self.df2 = pd.read_excel(file2_path)

    def get_patient_label(self, patientName):
        row1 = self.search_in_file(self.df1, 'patient', patientName)
        if row1 is not None:
            return self.interpret_file1_row(row1)

        row2 = self.search_in_file(self.df2, 'file', patientName)
        if row2 is not None:
            return self.interpret_file2_row(row2)

        return None

    def search_in_file(self, df, column, value):
        match = df[df[column].astype(str) == str(value)]
        if not match.empty:
            return match.iloc[0]
        return None

    def interpret_file1_row(self, row):
        analgesics = str(row.get('analgesics', '')).strip().lower()
        nrs = str(row.get('NRS(now)', '')).strip().lower()
        if analgesics == 'af':
            return 0
        elif nrs == '0' or nrs == '1' or nrs == '2' or nrs == '3' or nrs == '<3' :
            return 0
        else:
            return 1

    def interpret_file2_row(self, row):
        analgesics = str(row.get('analgesics', '')).strip().lower()
        nrs = str(row.get('(現在)NRS_now', '')).strip().lower()
        if analgesics.startswith('(non)') | analgesics.startswith('no') | analgesics.startswith('non') :
            return 0
        elif nrs == '0' or nrs == '1' or nrs == '2' or nrs == '3':
            return 0
        else:
            return 1


if __name__ == "__main__" : 
    
    Selector = PatientSelector()
    #print(Selector.get('NRS(now)', "0"))
    print("=====================================================")
    print(Selector.get_patient_label('P4_2011'))