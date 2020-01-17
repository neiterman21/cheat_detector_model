#!/usr/bin/env python

import csv
from shutil import copyfile
from pathlib import Path
import xml.etree.ElementTree as ET

DATA_PATH = "/home/evgeny/code_projects/cheat_detector_model/data/CheatGameLogs"
DeceptionDB_path = "/home/evgeny/code_projects/cheat_detector_model/data/DeceptionDB/"
#DeceptionDB_csv_path = "/home/evgeny/code_projects/cheat_detector_model/data/DeceptionDB/description.csv"
DeceptionDB_csv_path = "/home/evgeny/code_projects/cheat_detector_model/data/description.csv"

player_names = []
players = []

def ParsXML(file):
    tree = ET.parse(file)
    return tree.getroot()[-1][-1][-1].attrib

def ParsXMLgetPlayers(file , player_idx):
    tree = ET.parse(file)
    return tree.getroot()[0][player_idx].attrib

def print_stats():
    total_players = 0
    male = 0
    students = 0
    us = 0
    agv_age = 0
    education = {
        "Highschool" : 0,
        "BSc"        : 0,
        "MSc"        : 0,
        "Phd"        : 0
    }
    for player in players:
        total_players += 1
        if player["Gender"] == "Male":
            male += 1
        if player["IsStudent"] == "True":
            students += 1
        if player["CountryOfBirth"] == "United_States":
            us += 1
        agv_age += int(player["Age"])
        education[player["EducationType"]] +=1
    agv_age = agv_age/total_players

    print(total_players , male , students , us , agv_age)
    print(education)


def main():
    file_counter = 0
    with open(DeceptionDB_csv_path, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',  quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(["file_name",'IsTrueClaim'])
        for audio_filename in Path(DATA_PATH).glob('*/*/*/*/*.wav'):
            file_counter+=1
            description_path = Path(Path(audio_filename).parent).glob('*.xml').__next__()

            try:
                turn_attrib = ParsXML(description_path)
                player_attrib = ParsXMLgetPlayers(description_path , int(turn_attrib['PlayerIndex']))
              #  print(player_attrib)
                spamwriter.writerow(["claim_" + str(file_counter) + ".wav",turn_attrib['IsTrueClaim' ] , player_attrib])
            except:
                print (audio_filename)
            if (not (player_attrib["Name"] in player_names) ) :
                player_names.append(player_attrib["Name"])
                players.append(player_attrib)
              #  print(player_attrib)
           # copyfile(audio_filename, DeceptionDB_path + "claim_" +  str(file_counter) + ".wav")

    print("Total audio samples is: " + str(file_counter))
    print_stats()

if __name__ == "__main__":
    main()
