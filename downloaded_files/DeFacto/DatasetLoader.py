import os
import codecs
import pickle
import json
import numpy as np
import jsonlines
import datetime
from numpy import random


import doc_retrieval
import sentence_retrieval

random.seed(12345)

# global variable definition

label_fever_to_snli = {
	"SUPPORTS": "entailment",
	"REFUTES": 'contradiction',
	"NOT ENOUGH INFO": 'neutral'}


feverData_dir= "data/"
feverData_snliFormat_dir= "data/snli_format/"
train_filename = "train"
test_filename = "test"
dev_filename = "dev"

wiki_dir = 'data/wiki-pages/wiki-pages'


train_set = []
test_set = []
dev_set = []


def createDatasetRTEStyle(datasetFilename, testSetCreation= False):
	
	# Dev set 
	print("Processing "+ datasetFilename + "data ...\n")
	
	dev_file = jsonlines.open(feverData_dir + "/" + datasetFilename + ".jsonl")
	
	dev_set_snliFormat= codecs.open(feverData_snliFormat_dir + "/" + datasetFilename + "_snliFormat.jsonl", mode= "w")
	
	for line in dev_file:
		
		if (line["label"] == "SUPPORTS") or (line["label"] == "REFUTES"):
			
			evidencesList= []
			
			# during debug, I found examples where exactly the same evidence is provided multiple times for a claim. Adding this to the RTE system means that several repetitions will occur in the dataset. 
			# before adding to the dataset, check if evidence content was already added for this claim
			evidenceContentAdded= []
			
			# Add "entailment" or "contradiction" examples
			for evidenceSet in line["evidence"]:
				
				evidenceSetList= []
				
				for evidence in evidenceSet:
					# evidence is a list of the form: ["annotationId", "evidenceId", "docId", "sentenceId"]
					docIdEncoded= evidence[2].replace("/","-SLH-")
					docIdEncoded= docIdEncoded.encode('utf8').decode('utf8')
					currentDocContent= doc_retrieval.getDocContentFromFile(wiki_dir, docIdEncoded)  #doc_retrieval.getDocContent(wiki_dir, evidence[2])
					if currentDocContent is None:
						print("[ERROR] Document with id: ")
						print(evidence[2])
						print(" not found in the followinng directory: " + wiki_dir)
						print("")
					else:
						#currentDocContent= doc_retrieval.preProcessDoc(currentDocContent)
						evidenceContent= currentDocContent["lines"][evidence[3]]["content"]
						evidenceSetList.append({"docId": evidence[2], "evidenceId": evidence[3], "evidenceContent": evidenceContent})
					
				
				if len(evidenceSetList) > 0:
					
					"""
					for evi in evidenceSetList:
						print("claim: " + line["claim"] + " -> evidence id: " + str(evi["evidenceId"]) + "; content: " + evi["evidenceContent"] + "; docId: " + str(evi["docId"]) + "; fileId: " + str(evi["fileId"]))
					print("")
					"""
					
					# multiple evidences provided to support/refute claim
					# TODO: can be improved in many ways
					
					# current version: concatenates the text of all evidences. One learning instance is created (even if many evidences are provided)
					# motivation: if given separated (one example per evidence) the learning instance might not have enough info for the entailment or contradiction
					
					finalEvidenceContent= " ".join([evidence["evidenceContent"] for evidence in evidenceSetList])
					
					#NOTE: the "docId" field corresponds to the ids of the evidences used to support/contradict the claim separated by ";" (keeping the order of the evidences).
					finalEvidenceDocIds= ";".join([evidence["docId"] for evidence in evidenceSetList])
					
					evidenceRepetition= [0 for evidenceContentAlreadyAdded in evidenceContentAdded if evidenceContentAlreadyAdded == finalEvidenceContent]
					
					if len(evidenceRepetition) == 0:
						
						if testSetCreation:
							json.dump({"premise": finalEvidenceContent, "hypothesis": line["claim"], "gold_label": label_fever_to_snli[line["label"]], "docId": finalEvidenceDocIds}, dev_set_snliFormat)
						else:
							json.dump({"sentence1": finalEvidenceContent, "sentence2": line["claim"], "gold_label": label_fever_to_snli[line["label"]], "docId": finalEvidenceDocIds}, dev_set_snliFormat)
						dev_set_snliFormat.write("\n")
						evidenceContentAdded.append(finalEvidenceContent)
						
						evidencesList.append(evidenceSetList)
					
			
			# Add "none" example
			#TODO: can be improved in many ways
			
			# current version: adds the same number of examples as entailment/refutes previously added. Retrieve randomly one sentence from the same doc (different from the ones previously added, of course) as a negative example
			
			noneExamplesAdded= []
			
			for evidenceSetIndex in range(len(evidencesList)):
				
				# simplification: in case of multiple evidences, determine the pair doc/sentence only based on the first one in the list
				positiveExample= evidencesList[evidenceSetIndex][0]
				
				# list of prohibited sentences in doc (already added as positive example)
				sentenceIds= []
				sentenceIds.append(positiveExample["evidenceId"])
				
				# check all the positive (supports or contradictions) previously added -> goal: avoid a None example with exactly the same content as a previously added positive example
				for evidenceSetIndexAux in range(len(evidencesList)):
					for evidence in evidencesList[evidenceSetIndexAux]:
						if (not (evidenceSetIndexAux == evidenceSetIndex)) and (evidence["docId"] == positiveExample["docId"]):
							sentenceIds.append(evidence["evidenceId"])
				
				# add sentence idxs from already added none examples in the same doc
				for noneExample in noneExamplesAdded:
					if noneExample["docId"] == positiveExample["docId"]:
						sentenceIds.append(noneExample["evidenceId"])
				
				docIdEncoded= positiveExample["docId"].replace("/","-SLH-")
				docIdEncoded= docIdEncoded.encode('utf8').decode('utf8')
				currentDocContent= doc_retrieval.getDocContentFromFile(wiki_dir, docIdEncoded) #doc_retrieval.getDocContentFromFile(wiki_dir, positiveExample["fileId"], positiveExample["docId"])
				
				possibleSentenceIndexes= []
				
				if currentDocContent is None:
					print("[ERROR] Document with id= ")
					print(positiveExample["docId"])
					print(" not found!")
					print("")
				else:
					#currentDocContent= doc_retrieval.preProcessDoc(currentDocContent)
					possibleSentenceIndexes= list(set(sentenceIds).symmetric_difference(range(len(currentDocContent["lines"]))))
				
				
				if len(possibleSentenceIndexes) > 0:
					threshold= 0
					foundValidEvidence= False
					for possibleSentenceIndex in possibleSentenceIndexes:
						if len(currentDocContent["lines"][possibleSentenceIndex]["content"]) == 0:
							threshold= threshold + 1
						else:
							foundValidEvidence= True
							break
					
					#print("claim: " + line["claim"] + " -> evidence id: " + str(currentDocContent["lines"][possibleSentenceIndexes[0]]["evidenceId"]) + "; content: " + currentDocContent["lines"][possibleSentenceIndexes[0]]["content"] + "; docId: " + str(currentDocContent["lines"][possibleSentenceIndexes[0]]["docId"]) + "; fileId: " + str(currentDocContent["lines"][possibleSentenceIndexes[0]]["fileId"]))
					if foundValidEvidence:
						if testSetCreation:
							json.dump({"premise": currentDocContent["lines"][possibleSentenceIndexes[threshold]]["content"], "hypothesis": line["claim"], "gold_label": "neutral", "docId": positiveExample["docId"]}, dev_set_snliFormat)
						else:
							json.dump({"sentence1": currentDocContent["lines"][possibleSentenceIndexes[threshold]]["content"], "sentence2": line["claim"], "gold_label": "neutral", "docId": positiveExample["docId"]}, dev_set_snliFormat)
						dev_set_snliFormat.write("\n")
						noneExamplesAdded.append({"evidenceId": possibleSentenceIndexes[threshold], "docId": positiveExample["docId"]})
				
			
			
		
	
	dev_set_snliFormat.close()
	dev_file.close()
	

def randomUndersample():
	
	
	train_file = jsonlines.open(feverData_snliFormat_dir + "/" + train_filename + "_fever_snliFormat.jsonl")
	
	train_file_randomUndersample= codecs.open(feverData_snliFormat_dir + "/" + train_filename + "_fever_randomUndersample_snliFormat.jsonl", mode= "w")
	
	entailExamples= []
	contradictionExamples= []
	noneExamples= []
	
	for line in train_file:
		
		if line["gold_label"] == "entailment":
			entailExamples.append(line)
		elif line["gold_label"] == "contradiction":
			contradictionExamples.append(line)
		else:
			noneExamples.append(line)
		
	
	labelMinExamples= min(len(entailExamples), len(contradictionExamples), len(noneExamples))
	
	print(len(entailExamples))
	print(len(contradictionExamples))
	print(len(noneExamples))
	
	print("")
	
	random.shuffle(entailExamples)
	entailExamples= entailExamples[:labelMinExamples]
	
	random.shuffle(contradictionExamples)
	contradictionExamples= contradictionExamples[:labelMinExamples]
	
	random.shuffle(noneExamples)
	noneExamples= noneExamples[:labelMinExamples]
	
	print(len(entailExamples))
	print(len(contradictionExamples))
	print(len(noneExamples))
	
	dataset= entailExamples + contradictionExamples + noneExamples
	random.shuffle(dataset)
	
	print(len(dataset))
	
	for elem in dataset:
		json.dump(elem, train_file_randomUndersample)
		train_file_randomUndersample.write("\n")
	
	train_file_randomUndersample.close()
	train_file.close()

"""
print("Starting dev set")
print(datetime.datetime.now())
createDatasetRTEStyle(dev_filename)

print("Created dev data successfully!")
print(datetime.datetime.now())

print("Starting train set")
print(datetime.datetime.now())
createDatasetRTEStyle(train_filename)

print("Starting test set")
print(datetime.datetime.now())
createDatasetRTEStyle(test_filename, testSetCreation= True)
"""

#randomUndersample()
