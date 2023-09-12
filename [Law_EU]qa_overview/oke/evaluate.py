import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import json
import numpy as np

from more_itertools import unique_everseen
import itertools

import sys
import logging
logger = logging.getLogger('knowpy')
logger.setLevel(logging.INFO)
# logger.setLevel(logging.ERROR)

global model_type
global info_type
_,top_k,tfidf_importance,info_type,document_based_eval,model_type,log_dir = sys.argv
tfidf_importance = float(tfidf_importance)
top_k = int(top_k)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

model_type = model_type.casefold()
info_type = info_type.casefold()
print(f'evaluate {info_type} {model_type}')
from server_interface import *
qa = init(info_type,model_type,tfidf_importance)

DOCUMENT_BASED_EVAL = document_based_eval.casefold()=='true'

question_evaluation_score = {}
question_dict = {
	"How is a qualified electronic signature validated?": {
		"specificity": "H",
		"documents": ['E'],
		"expected_answers": [
			'E Art. 32',
			'E Art. 33',
			'E Rec. 57',
		]
	},
	"Can an electronic signature be expressed in the form of a pseudonym?": {
		"specificity": "N",
		"documents": ['E'],
		"expected_answers": [
			'E Art. 3.14',
			'E Art. 32',
		]
	},
	"Can a minor obtain a qualified electronic signature?": {
		"specificity": "L",
		"documents": ['E'],
		"expected_answers": [
			'E Art. 3',
			'E Art. 25',
		]
	},
	"From when qualified certificates lose their validity in the case of revocation?": {
		"specificity": "N",
		"documents": ['E'],
		"expected_answers": [
			'E Art. 24',
			'E Art. 28',
		]
	},
	"Is a graphometric signature qualified as an advanced electronic signature?": {
		"specificity": "L",
		"documents": ['E'],
		"expected_answers": [
			'E Art. 3.11',
			'E Art. 26',
		]
	},
	"How should access to trust services be granted to persons with disabilities?": {
		"specificity": "N",
		"documents": ['E'],
		"expected_answers": [
			'E Rec. 29',
			'E Art. 15',
		]
	},
	"How can the identity of a natural person be verified in the issuing of a qualified certificate?": {
		"specificity": "H",
		"documents": ['E'],
		"expected_answers": [
			'E Art. 24.1',
		]
	},
	"Do electronic contracts have the same validity as paper contracts?": {
		"specificity": "L",
		"documents": ['E'],
		"expected_answers": [
			'E Rec. 21',
			'E Art. 2.3',
		]
	},
	"Why is there a specific discipline for the notification of security breaches?": {
		"specificity": "H",
		"documents": ['E'],
		"expected_answers": [
			'E Rec. 38',
			'E Art. 19.2',
		]
	},
	"When shall a trust service provider notify affected individuals and users?": {
		"specificity": "H",
		"documents": ['E'],
		"expected_answers": [
			'E Art. 19.2',
		]
	},
	"What is the applicable law to the trust service provider which provides its trusted services in a Member State different from the one where it is established?": {
		"specificity": "L",
		"documents": ['E'],
		"expected_answers": [
			'E Rec. 22',
			'E Rec. 42',
			'E Art. 4',
			'E Art. 6',
			'E Art. 24',
		]
	},
	"How can qualified certificates be temporally limited?": {
		"specificity": "N",
		"documents": ['E'],
		"expected_answers": [
			'E Rec. 53',
			'E Art. 24.4',
			'E Art. 28',
			'E Art. 38.5',
		]
	},
	"What are the requirements for website authentication?": {
		"specificity": "N",
		"documents": ['E'],
		"expected_answers": [
			'E Rec. 67',
			'E Art. 45',
		]
	},
	'When do electronic signatures qualify as "advanced electronic signatures"?': {
		"specificity": "N",
		"documents": ['E'],
		"expected_answers": [
			'E Art. 3.11',
			'E Art. 26',
		]
	},
	"Which subject has the competence to maintain trusted lists?": {
		"specificity": "H",
		"documents": ['E'],
		"expected_answers": [
			'E Art. 22',
		]
	},
	"How should liability be determined for Member States that are non-compliant with provisions about electronic identification schemes?": {
		"specificity": "N",
		"documents": ['E'],
		"expected_answers": [
			'E Rec. 18',
			'E Art. 11',
		]
	},
	"What is a security breach?": {
		"specificity": "L",
		"documents": ['E'],
		"expected_answers": [
			'E Art. 10',
			'E Art. 19',
		]
	},
	#################################################################################
	#################################################################################
	"When is it mandatory to carry out a Data Protection Impact Assessment?": {
		"specificity": "H",
		"documents": ['G'],
		"expected_answers": [
			'G Art. 35.1',
			'G Art. 35.3',
		]
	},
	"What are the possible security measures that can be adopted to mitigate the risks related to personal data processing?": {
		"specificity": "N",
		"documents": ['G'],
		"expected_answers": [
			'G Art. 32.1',
			'G Art. 32.2',
		]
	},
	"What are the applicable rules to the processing of personal data for archiving purposes in the public interest, for scientific or historical research purposes or for statistical purposes?": {
		"specificity": "L",
		"documents": ['G'],
		"expected_answers": [
			'G Rec. 156',
			'G Art. 5.1.b',
			'G Art. 9.2.j',
			'G Art. 14.5.b',
			'G Art. 17.3.d',
			'G Art. 89',
		]
	},
	"How should a data processor be appointed?": {
		"specificity": "N",
		"documents": ['G'],
		"expected_answers": [
			'G Art. 26',
			'G Art. 38',
		]
	},
	"When is the consent of the data subject explicit?": {
		"specificity": "L",
		"documents": ['G'],
		"expected_answers": [
			'G Rec. 51',
			'G Rec. 71',
			'G Rec. 111',
			'G Art. 7.1',
			'G Art. 9',
		]
	},
	"What elements shall the European Commission keep into account to authorise the transfer of personal data to a third country through an Adequacy Decision?": {
		"specificity": "N",
		"documents": ['G'],
		"expected_answers": [
			'G Art. 45.2',
			'G Art. 45.3',
			'G Rec. 104',
		]
	},
	"What are the rules applicable to biometric data?": {
		"specificity": "H",
		"documents": ['G'],
		"expected_answers": [
			'G Rec. 51',
			'G Rec. 53',
			'G Art. 9',
		]
	},

	"When does the public interest override data subject rights?": {
		"specificity": "L",
		"documents": ['G'],
		"expected_answers": [
			'G Rec. 45',
			'G Rec. 46',
			'G Rec. 50',
			'G Rec. 65',
			'G Rec. 69',
			'G Art. 9.2.i',
			'G Art. 17.3',
			'G Art. 89',
		]
	},
	"To what data is the right to portability applicable?": {
		"specificity": "H",
		"documents": ['G'],
		"expected_answers": [
			'G Art. 20',
		]
	},
	"How should a data processing record be drafted?": {
		"specificity": "H",
		"documents": ['G'],
		"expected_answers": [
			'G Art. 30',
		]
	},
	"What data processing poses significant risks to the fundamental rights and freedoms of natural persons?": {
		"specificity": "N",
		"documents": ['G'],
		"expected_answers": [
			'G Rec. 51',
			'G Rec. 75',
			'G Art. 9',
			'G Art. 10'
		]
	},
	"What elements should be included in a Code of Conduct?": {
		"specificity": "N",
		"documents": ['G'],
		"expected_answers": [
			'G Rec. 81',
			'G Art. 40',
		]
	},
	"What are the obligations of the data controller when the legal basis for the data processing is the consent of the data subject?": {
		"specificity": "N",
		"documents": ['G'],
		"expected_answers": [
			'G Art. 7',
			'G Art. 13',
			'G Art. 14',
			'G Art. 20'
		]
	},
	"Which legal entity can impose fines on data controllers?": {
		"specificity": "H",
		"documents": ['G'],
		"expected_answers": [
			'G Rec. 130',
			'G Art. 58.2.i',
			'G Art. 83',
		]
	},
	"Who can exercise the right to lodge a complaint before the supervisory authority?": {
		"specificity": "N",
		"documents": ['G'],
		"expected_answers": [
			'G Rec. 141',
			'G Rec. 142',
			'G Art. 77',
		]
	},
	"What is the procedure to follow in the event of a data breach?": {
		"specificity": "L",
		"documents": ['G'],
		"expected_answers": [
			'G Rec. 85',
			'G Rec. 86',
			'G Rec. 87',
			'G Rec. 88',
			'G Art. 33',
			'G Art. 34'
		]
	},
	#################################################################################
	#################################################################################
	"Who determines disputes under a contract?": {
		"specificity": "L",
		"documents": ['B'],
		"expected_answers": [
			"B Art. 7.1",
			"B Art. 8.3",
			"B Art. 8.4",
			"B Art. 17"
		]
	},
	"What factors should be taken into account for conferring the jurisdiction to determine disputes under a contract?": {
		"specificity": "N",
		"documents": ['B'],
		"expected_answers": [
			"B Art. 7.1",
			"B Art. 17",
			"B Art. 20",
			"B Art. 25"
		]
	},
	"Which parties of a contract should be protected by conflict-of-law rules?": {
		"specificity": "N",
		"documents": ['RI'],
		"expected_answers": [
			"RI Rec. 23",
			"RI Art. 6",
			"RI Art. 8",
			"RI Art. 13"
		]
	},
	"In which case are claims so closely connected that it would be better to treat them together in order to avoid irreconcilable judgments?": {
		"specificity": "H",
		"documents": ['B'],
		"expected_answers": [
			"B Art. 8",
			"B Art. 30",
			"B Art. 34"
		]
	},
	"In which court is celebrated the trial in case the employer is domiciled in a Member State?": {
		"specificity": "H",
		"documents": ['B'],
		"expected_answers": [
			"B Art. 21",
			"B Art. 22",
			"B Art. 23"
		]
	},
	"Which law is applicable to a non-contractual obligation?": {
		"specificity": "N",
		"documents": ['RII'],
		"expected_answers": [
			"RII Rec. 17",
			"RII Rec. 18",
			"RII Rec. 26",
			"RII Rec. 27",
			"RII Rec. 31",
			"RII Art. 4",
			"RII Art. 5",
			"RII Art. 6",
			"RII Art. 7",
			"RII Art. 8",
			"RII Art. 9",
			"RII Art. 10",
			"RII Art. 11",
			"RII Art. 12",
			"RII Art. 13",
			"RII Art. 14",
			"RII Art. 15",
			"RII Art. 16",
			"RII Art. 17",
			"RII Art. 18",
			"RII Art. 19",
			"RII Art. 20"
		]
	},
	"Can the parties choose the applicable law in consumer contracts?": {
		"specificity": "H",
		"documents": ['RI'],
		"expected_answers": [
			"RI Rec. 11",
			"RI Rec. 25",
			"RI Rec. 27",
			"RI Art. 6"
		]
	},
	"What factors should be taken into account for conferring the jurisdiction to determine disputes under a consumer contract?": {
		"specificity": "N",
		"documents": ['B'],
		"expected_answers": [
			"B Rec. 18",
			"B Art. 17",
			"B Art. 18",
			"B Art. 19",
			"B Art. 26"
		]
	},
	"Can the parties choose a different applicable law for different parts of the contract?": {
		"specificity": "L",
		"documents": ['RI'],
		"expected_answers": [
			"RI Rec. 11",
			"RI Art. 3.1"
		]
	},
	"What is the applicable rule to protect the weaker party of a contract?": {
		"specificity": "N",
		"documents": ['B','RI'],
		"expected_answers": [
			"RI Rec. 23",
			"B Rec. 18"
		]
	},
	"What is the applicable law to determine the validity of consent?": {
		"specificity": "L",
		"documents": ['RI'],
		"expected_answers": [
			"RI Art. 3.5",
			"RI Art. 10",
			"RI Art. 11",
			"RI Art. 13"
		]
	},
	"What court has jurisdiction in case of a counter-claim?": {
		"specificity": "N",
		"documents": ['B'],
		"expected_answers": [
			"B Art. 8.3",
			"B Art. 14.2",
			"B Art. 18.3",
			"B Art. 22.2"
		]
	},
	"Where can an employee sue their employer?": {
		"specificity": "H",
		"documents": ['B'],
		"expected_answers": [
			"B Rec. 14",
			"B Rec. 18",
			"B Art. 21.1",
			"B Art. 22.1",
			"B Art. 23"
		]
	},
	#################################################################################
	#################################################################################
	"What is the European arrest warrant?": {
		"specificity": "N",
		"documents": ['W'],
		"expected_answers": [
			'W Art. 1.1',
			'W Art. 8',
			'W Art. 9.3',
			'W Rec. 11',
			'W Rec. 6',
		]
	},
	"Can the execution of the European arrest warrant be refused when the law of the executing Member State does not impose the same type of tax or duty or does not contain the same type of tax rules as the law of the issuing Member State?": {
		"specificity": "L",
		"documents": ['W'],
		"expected_answers": [
			'W Art. 2.2', 
			'W Art. 2.4',
			'W Art. 4.1',
			'W Rec. 6',
		]
	},
	"Who decides precedence in the event of a conflict between a European arrest warrant and a request for extradition from a third country?": {
		"specificity": "N",
		"documents": ['W'],
		"expected_answers": [
			'W Art. 16.3', 
			'W Rec. 8',
			'W Art. 10.6',
		]
	},
	"Which law is used to record the consent to surrender of a requested person?": {
		"specificity": "H",
		"documents": ['W'],
		"expected_answers": [
			'W Art. 13.3',
			'W Art. 11',
		]
	},
	"Is the arrest warrant based on the principle of mutual recognition?": {
		"specificity": "L",
		"documents": ['W'],
		"expected_answers": [
			'W Rec. 2', 
			'W Rec. 6', 
			'W Rec. 5', 
			'W Art. 1.1', 
			'W Art. 1.2', 
			'W Rec. 10', 
		]
	},
	"Does a requested person have the right to an interpreter?": {
		"specificity": "H",
		"documents": ['W'],
		"expected_answers": [
			'W Art. 11.2', 
		]
	},
	"Can the consent to the surrender of the arrested person be revoked?": {
		"specificity": "N",
		"documents": ['W'],
		"expected_answers": [
			'W Art. 13.4', 
			'W Art. 17', 
		]
	},
	"Is the surrender of the arrested person always subject to the verification of the double criminality of the act?": {
		"specificity": "L",
		"documents": ['W'],
		"expected_answers": [
			'W Art. 2.2', 
			'W Art. 2.3', 
			'W Art. 2.4',
			'W Art. 4.1',
			'W Art. 5', 
			'W Art. 33',
		]
	},
	"Which authority should be informed in case of repeated delays by a Member State in executing European arrest warrants?": {
		"specificity": "H",
		"documents": ['W'],
		"expected_answers": [
			'W Art. 17.7', 
		]
	},
	"Can the Member States also apply other agreements in addition to the Framework Decision?": {
		"specificity": "L",
		"documents": ['W'],
		"expected_answers": [
			'W Art. 31', 
			'W Rec. 5', 
			'W Art. 33', 
			'W Art. 32',
		]
	},
	"Can the European arrest warrant be ordered for the execution of a non-custodial sentence?": {
		"specificity": "N",
		"documents": ['W'],
		"expected_answers": [
			'W Art. 2.1', 
			'W Art. 1.1', 
			'W Rec. 12', 
			'W Art. 5',
		]
	},
	"Can the executing judicial authority refuse to execute the European arrest warrant when the person who is the subject of the European arrest warrant is being prosecuted in the executing Member State for the same act as that on which the European arrest warrant is based?": {
		"specificity": "N",
		"documents": ['W'],
		"expected_answers": [
			'W Art. 4.2',
			'W Rec. 8', 
			'W Art. 24', 
			'W Rec. 13',
		]
	},
	"What right is applied by the judicial authority to decide whether the requested person should remain in detention or be provisionally released?": {
		"specificity": "H",
		"documents": ['W'],
		"expected_answers": [
			'W Art. 12.1', 
			'W Rec. 8', 
			'W Rec. 10', 
		]
	},
	"Can the constitutional rules of the Member States be applied?": {
		"specificity": "L",
		"documents": ['W'],
		"expected_answers": [
			'W Rec. 7', 
			'W Rec. 12', 
			'W Art. 1.3', 
			'W Art. 34',
		]
	},
	"Should the European arrest warrant be translated into the official language or one of the official languages of the executing Member State?": {
		"specificity": "H",
		"documents": ['W'],
		"expected_answers": [
			'W Art. 8.2',
			'W Rec. 8',
		]
	},
	"Can the executing judicial authority request the opinion of Eurojust in case of multiple requests?": {
		"specificity": "H",
		"documents": ['W'],
		"expected_answers": [
			'W Art. 16.2', 
			'W Rec. 8', 
		]
	},
	"Can the executing judicial authority, on its own initiative, seize and hand over property acquired by the requested person as a result of an offence?": {
		"specificity": "N",
		"documents": ['W'],
		"expected_answers": [
			'W Art. 29.1', 
			'W Rec. 5', 
		]
	},
	"Is an alert in the Schengen Information System equivalent to a European arrest warrant?": {
		"specificity": "N",
		"documents": ['W'],
		"expected_answers": [
			'W Art. 9.3', 
			'W Art. 8.1', 
			'W Art. 1.1', 
		]
	},
	"What are the time limits for the surrender of the requested person?": {
		"specificity": "L",
		"documents": ['W'],
		"expected_answers": [
			'W Art. 23', 
			'W Art. 15', 
			'W Art. 17', 
			'W Art. 20', 
			'W Art. 24',
			'W Rec. 1', 
		]
	},
	"How are the expenses of executing the European arrest warrant allocated?": {
		"specificity": "H",
		"documents": ['W'],
		"expected_answers": [
			'W Art. 30', 
		]
	},
	"What claims can be made to the judicial authority by the interested party who has not previously received any official information on the existence of the criminal proceedings against him/her?": {
		"specificity": "L",
		"documents": ['W'],
		"expected_answers": [
			'W Art. 4a', 
			'W Rec. 12', 
			'W Art. 11',
		]
	},
}
# if DOCUMENT_BASED_EVAL:
question_dict.update({
	"What kind of agreement between parties is regulated by these Regulations?": {
		"specificity": "L",
		"documents": ['B','RI','RII'],
		"expected_answers": [
			"B Rec. 6",
			"B Rec. 10",
			"B Rec. 12",
			"B Art. 1",
			"RI Rec. 7",
			"RI Art. 1"
		]
	},
	"How should a contract be interpreted according to Regulation Rome I?": {
		"specificity": "L",
		"documents": ['RI'],
		"expected_answers": [
			"RI Rec. 22",
			"RI Rec. 12",
			"RI Rec. 26",
			"RI Rec. 29",
			"RI Art. 12"
		]
	},
	"What non-contractual obligations fall into the scope of Regulation Rome II?": {
		"specificity": "H",
		"documents": ['RII'],
		"expected_answers": [
			"RII Rec. 10",
			"RII Rec. 11",
			"RII Art. 1",
			"RII Art. 2"
		]
	},
	"When are two actions to be considered related according to the Regulation Brussels I Bis?": {
		"specificity": "N",
		"documents": ['B'],
		"expected_answers": [
			"B Rec. 21",
			"B Art. 30.3"
		]
	},	
	"Does the GDPR provide a right to explanation?": {
		"specificity": "L",
		"documents": ['G'],
		"expected_answers": [
			'G Rec. 71',
			'G Art. 12.3',
			'G Art. 15.1'
		]
	},
})

def format_answer_id(answer_id):
	# print(answer_id)
	# print(answer_id.replace(u'\xa0', ' ').split(' '))
	d,t,n = answer_id.replace(u'\xa0', ' ').split(' ')
	n = n.split('.')[0]
	return ' '.join([d,t,n])

def get_top_k_precision(given_answers, expected_answers, top_k):
	# given_answers = unique_everseen(given_answers)
	given_answers = itertools.islice(given_answers, top_k)
	return len(set(given_answers).intersection(expected_answers))/min(top_k,len(expected_answers)) if expected_answers else 0

def get_top_k_recall(given_answers, expected_answers, top_k):
	# expected_answers = set(expected_answers)
	return sum((1 if ga in expected_answers else 0 for ga in given_answers[:top_k]))/min(top_k,len(given_answers)) if given_answers else 0

def get_f1_score(p,r):
	return 2*p*r/(p+r) if p+r else 0

def get_discounted_cumulative_gain(given_answers, expected_answers, top_k):
	return sum(map(lambda x: (1 if x[1] in expected_answers else 0)/np.log2(x[0]+2), enumerate(given_answers[:top_k]))) if given_answers else 0

def get_ideal_discounted_cumulative_gain(given_answers, expected_answers, top_k):
	sorted_given_answers = sorted(given_answers, key=lambda x: 1 if x in expected_answers else 0, reverse=True)
	return get_discounted_cumulative_gain(sorted_given_answers, expected_answers, top_k)

def get_normalised_discounted_cumulative_gain(given_answers, expected_answers, top_k):
	idcg = get_ideal_discounted_cumulative_gain(given_answers, expected_answers, top_k)
	return get_discounted_cumulative_gain(given_answers, expected_answers, top_k)/idcg if idcg else 0

def get_reciprocal_rank(given_answers, expected_answers, top_k):
	first_correct_answer = next(filter(lambda x: x[-1] in expected_answers, enumerate(given_answers[:top_k])),None)
	if first_correct_answer is None:
		return 0
	return 1/(first_correct_answer[0]+1)

for question, info_dict in question_dict.items():
	question_answer_dict = get_question_answer_dict(
		qa,
		[question],
		options=OQA_OPTIONS
	)
	if not question_answer_dict:
		question_evaluation_score[question] = {
			'precision': 0,
			'recall': 0,
			'f1': 0,
			'ndcg': 0,
			'rr': 0,
		}
		continue

	answer_id_list = []
	for answer in unique_everseen(question_answer_dict[question.strip()], key=lambda x: x['sentence']):
		triplet_list = answer['extra_info']

		article_id = next(map(lambda x:x[-1], filter(lambda x: x[1] == 'my:article_id', triplet_list)),None)
		if article_id:
			article_id = article_id.strip(')').strip('(')
			article_id = article_id.replace('Article','Art.')
			if not article_id.startswith('Art.'):
				article_id = 'Art. '+article_id

		recital_id = next(map(lambda x:x[-1], filter(lambda x: x[1] == 'my:recital_id', triplet_list)),None)
		if recital_id:
			recital_id = recital_id.strip(')').strip('(')
			recital_id = recital_id.replace('Recital','Rec.')
			if not recital_id.startswith('Rec.'):
				recital_id = 'Rec. '+recital_id

		if not article_id and not recital_id:
			continue

		doc_id = next(map(lambda x:x[-1], filter(lambda x: x[1] == 'my:docID', triplet_list)),None)
		if doc_id.startswith('myf:rome_ii'):
			doc_id = 'RII'
		elif doc_id.startswith('myf:rome_i'):
			doc_id = 'RI'
		elif doc_id.startswith('myf:bruss'):
			doc_id = 'B'
		elif doc_id.startswith('myf:gdpr'):
			doc_id = 'G'
		elif doc_id.startswith('myf:eidas'):
			doc_id = 'E'
		elif doc_id.startswith('myf:warrant'):
			doc_id = 'W'
		
		if DOCUMENT_BASED_EVAL and doc_id not in info_dict["documents"]:
			continue

		answer_id = ' '.join(filter(lambda x:x,[doc_id,article_id,recital_id]))
		answer_id_list.append(answer_id)

	# answer_id_list = list(unique_everseen(map(format_answer_id, answer_id_list)))
	answer_id_list = list(map(format_answer_id, answer_id_list))
	expected_answer_set = set(map(format_answer_id, info_dict['expected_answers']))
	p = get_top_k_precision(answer_id_list, expected_answer_set, top_k)
	r = get_top_k_recall(answer_id_list, expected_answer_set, top_k)
	f1 = get_f1_score(p,r)
	ndcg = get_normalised_discounted_cumulative_gain(answer_id_list, expected_answer_set, top_k)
	rr = get_reciprocal_rank(answer_id_list, expected_answer_set, top_k)
	question_evaluation_score[question] = {
		'precision': p,
		'recall': r,
		'f1': f1,
		'ndcg': ndcg,
		'rr': rr,
	}
	print(question, json.dumps(answer_id_list, indent=4))
store_cache(qa)

r3 = lambda x: round(x, 3)

final_results_dict = {}
final_results_dict['total_scores'] = question_evaluation_score

precision_list = [x['precision'] for x in question_evaluation_score.values()]
recall_list = [x['recall'] for x in question_evaluation_score.values()]
f1_list = [x['f1'] for x in question_evaluation_score.values()]
ndcg_list = [x['ndcg'] for x in question_evaluation_score.values()]
rr_list = [x['rr'] for x in question_evaluation_score.values()]

def build_stats_dict(x, label=''):
	return {
	'mean': f'{label}{r3(np.mean(x))} ± {r3(np.std(x))}',
	'median':f'{label}{r3(np.median(x))} <{r3(np.quantile(x, 0.25))}, {r3(np.quantile(x, 0.75))}>'
}

final_results_dict['precision'] = build_stats_dict(precision_list, 'P: ')
final_results_dict['recall'] = build_stats_dict(recall_list, 'R: ')
final_results_dict['f1'] = build_stats_dict(f1_list, 'F1: ')
final_results_dict['ndcg'] = build_stats_dict(ndcg_list, 'NDCG: ')
final_results_dict['rr'] = build_stats_dict(rr_list, 'RR: ')

specificity_dict = {
	'L': [],
	'N': [],
	'H': [],
}
for k,x in question_evaluation_score.items():
	specificity = question_dict[k]["specificity"]
	specificity_dict[specificity].append(x['precision'])

final_results_dict['specificity_dict'] = {
	specificity: build_stats_dict(score_list)
	for specificity, score_list in specificity_dict.items()
}

with open(os.path.join(log_dir,f'{model_type}.json'), 'w') as f:
	json.dump(final_results_dict, f, indent=4, ensure_ascii=False)
