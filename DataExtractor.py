#!/usr/bin/env python

import csv
from shutil import copyfile
from pathlib import Path
import xml.etree.ElementTree as ET

DATA_PATH = "/home/evgeny/code_projects/cheat_detector_model/data/"
DeceptionDB_path = "/home/evgeny/code_projects/cheat_detector_model/data/DeceptionDB/"
DeceptionDB_csv_path = "/home/evgeny/code_projects/cheat_detector_model/data/DeceptionDB/description.csv"

def ParsXML(file):
    tree = ET.parse(file)
    return tree.getroot()[1][-1][-1].attrib

def main():
    file_counter = 0
    with open(DeceptionDB_csv_path, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',  quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(["file_name",'IsTrueClaim'])
        for audio_filename in Path(DATA_PATH).glob('*/*/*/*/*/*.wav'):
            file_counter+=1
            description_path = Path(Path(audio_filename).parent).glob('*.xml').__next__()
            turn_attrib = ParsXML(description_path)
            spamwriter.writerow(["claim_" + str(file_counter) + ".wav",turn_attrib['IsTrueClaim']])
            copyfile(audio_filename, DeceptionDB_path + "claim_" +  str(file_counter) + ".wav")


    print("Total audio samples is: " + str(file_counter))

if __name__ == "__main__":
    main()
