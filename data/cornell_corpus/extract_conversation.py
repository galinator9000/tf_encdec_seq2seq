#! -*- coding: UTF-8 -*-
"""
Extract all conversations in Cornell corpus as TSV format.
"""

import re

exampleLimit = 50000
maxWord = 20

# Filter given list.
def apply_filter(x):
	return list(filter(None, x))

# Clean unwanted chars.
def clean_data(data):
	data = re.sub(r"""[^A-Za-z0-9 ]""", "", data)
	for i in range(0, 100):
		data = data.replace("  ", " ")
	data = data.replace("\n ", "\n").replace(" \n", "\n").replace("\t ", "\t").replace(" \t", "\t").replace(" '", "'").replace("' ", "'")
	return data.lower().strip()

# Reads given file, seperates with seperator and returns as list.
def read_and_split(fName, seperator):
	return [line.split(seperator) for line in apply_filter(open(fName, "r").read().split("\n"))]

movie_lines = read_and_split("movie_lines.txt", " +++$+++ ")
movie_conversations = read_and_split("movie_conversations.txt", " +++$+++ ")

# Build dictionaries that holds lines' text and characterID's
Ltext = {}
Lchar = {}
for line in movie_lines:
	lineID = line[0]
	charID = line[1]
	text = clean_data(line[-1])

	Ltext.update(
		{lineID:text}
	)
	Lchar.update(
		{lineID:charID}
	)

# Takes LineIDs from conversation.
movie_conversation_lines = []
for conversation in movie_conversations:
	movie_conversation_lines.append(
		re.findall(r"""\'(.*?)\'""", conversation[-1])
	)

all_dataF = open("all_data.txt", "w", encoding="utf-8")
exampleCount = 0

# Extracts texts from conversation. If pairs valid, writes them to file as TSV format.
for conversation in movie_conversation_lines:
	for lineIndex in range(0, len(conversation)-1):
		lineID_1 = conversation[lineIndex]
		lineID_2 = conversation[lineIndex+1]

		if Lchar[lineID_1] != Lchar[lineID_2] and len(Ltext[lineID_1].split(" ")) < maxWord and len(Ltext[lineID_2].split(" ")) < maxWord:
			all_dataF.write(
				Ltext[lineID_1] + "\t" + Ltext[lineID_2] + "\n"
			)
			exampleCount += 1

	if exampleCount >= exampleLimit:
		break
all_dataF.close()