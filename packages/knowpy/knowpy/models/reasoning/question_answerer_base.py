from knowpy.misc.doc_reader import DocParser

from knowpy.models.classification.concept_classifier import ConceptClassifier
from knowpy.models.classification.sentence_classifier import SentenceClassifier
from knowpy.models.summarisation.neural_sentence_summariser import NeuralSentenceSummariser
from knowpy.models.knowledge_extraction.question_answer_extractor import QuestionAnswerExtractor
from knowpy.models.knowledge_extraction.couple_extractor import filter_invalid_sentences

# from knowpy.misc.graph_builder import get_concept_description_dict
from knowpy.misc.levenshtein_lib import remove_similar_labels
from knowpy.misc.jsonld_lib import *
from knowpy.misc.utils import *
from knowpy.models.reasoning import is_not_wh_word

import numpy as np
from collections import Counter
import re
import time
import json
from more_itertools import unique_everseen
import itertools
import wikipedia
from collections import namedtuple
import logging
from knowpy.models.model_manager import ModelManager

ArchetypePertinence = namedtuple('ArchetypePertinence',['archetype','pertinence'])
InformationUnit = namedtuple('InformationUnit',['unit','context'])
# get_information_unit = lambda x: InformationUnit(x['abstract'], x['sentence'])

class QuestionAnswererBase(ModelManager):
	archetypal_questions_dict = {
		##### Descriptive
		# 'what': 'What is a description of {X}?',
		'what': 'What is {X}?',
		# 'what': 'What is {X}?',
		'who': 'Who is {X}?',
		# 'whom': 'Whom {X}?',
		##### Causal + Justificatory
		'why': 'Why {X}?',
		# 'why-not': 'Why not {X}?',
		##### Counterfactual
		# 'what-if': 'What if {X}?',
		##### Teleological
		# 'what-for': 'What is {X} for?',
		# 'what-for': 'What is {X} for?',
		##### Expository
		'how': 'How is {X}?',
		##### Quantitative
		# 'how-much': 'How much {X}?',
		# 'how-many': 'How many {X}?',
		##### Spatial
		'where': 'Where is {X}?',
		##### Temporal
		'when': 'When is {X}?',
		##### Medium
		# 'who-by': 'Who by {X}?',
		##### Extra
		'which': 'Which {X}?',
		'whose': 'Whose {X}?',
		##### Discourse Relations
		'Expansion.Manner': 'In what manner {X}?', # (25\%),
		'Contingency.Cause': 'What is the reason {X}?', # (19\%),
		'Contingency.Effect': 'What is the result of {X}?', # (16\%),
		'Expansion.Level-of-detail': 'What is an example of {X}?', # (11\%),
		'Temporal.Asynchronous.Consequence': 'After what {X}?', # (7\%),
		'Temporal.Synchronous': 'While what {X}?', # (6\%),
		'Contingency.Condition': 'In what case {X}?', # (3),
		'Comparison.Concession': 'Despite what {X}?', # (3\%),
		'Comparison.Contrast': 'What is contrasted with {X}?', # (2\%),
		'Temporal.Asynchronous.Premise': 'Before what {X}?', # (2\%),
		'Temporal.Asynchronous.Being': 'Since when {X}?', # (2\%),
		'Comparison.Similarity': 'What is similar to {X}?', # (1\%),
		'Temporal.Asynchronous.End': 'Until when {X}?', # (1\%),
		'Expansion.Substitution': 'Instead of what {X}?', # (1\%),
		'Expansion.Disjunction': 'What is an alternative to {X}?', # ($\leq 1\%$),
		'Expansion.Exception': 'Except when {X}?', # ($\leq 1\%$),
		'Contingency.Neg.-cond.': 'Unless what {X}?', # ($\leq 1\%$).
	}

	def __init__(self, kg_manager, concept_classifier_options, sentence_classifier_options, answer_summariser_options=None, betweenness_centrality=None, **args):
		super().__init__(sentence_classifier_options)
		self.disable_spacy_component = ["ner","textcat"]
		
		self.betweenness_centrality = betweenness_centrality
		self.kg_manager = kg_manager

		# Concept classification
		self.concept_classifier_options = concept_classifier_options
		self._concept_classifier = None
		# Sentence classification
		self.sentence_classifier_options = sentence_classifier_options
		self._sentence_classifier = None
		# Summarisation
		self.answer_summariser_options = answer_summariser_options
		self._sentence_summariser = None

		self._overview_aspect_set = None
		self._relevant_aspect_set = None

	@property
	def sentence_summariser(self):
		if not self._sentence_summariser and self.answer_summariser_options:
			self._sentence_summariser = NeuralSentenceSummariser(self.answer_summariser_options)
		return self._sentence_summariser

	@property
	def sentence_classifier(self):
		if self._sentence_classifier is None:
			self._sentence_classifier = SentenceClassifier(self.sentence_classifier_options)
			self._init_sentence_classifier()
		return self._sentence_classifier

	@property
	def concept_classifier(self):
		if self._concept_classifier is None:
			self._concept_classifier = ConceptClassifier(self.concept_classifier_options)
			self._init_concept_classifier()
		return self._concept_classifier

	@property
	def overview_aspect_set(self):
		if self._overview_aspect_set is None:
			self._overview_aspect_set = set(filter(lambda x: self.kg_manager.is_relevant_aspect(x,ignore_leaves=True), self.kg_manager.aspect_uri_list))
			# Betweenness centrality quantifies the number of times a node acts as a bridge along the shortest path between two other nodes.
			if self.betweenness_centrality is not None:
				filtered_betweenness_centrality = dict(filter(lambda x: x[-1] > 0, self.betweenness_centrality.items()))
				self._overview_aspect_set &= filtered_betweenness_centrality.keys()
		return self._overview_aspect_set

	@property
	def relevant_aspect_set(self):
		if self._relevant_aspect_set is None:
			self._relevant_aspect_set = set(filter(self.kg_manager.is_relevant_aspect, self.kg_manager.aspect_uri_list))
		return self._relevant_aspect_set

	@property
	def adjacency_list(self):
		return self.kg_manager.adjacency_list

	def get_formatted_answer(self, answer):
		triple, (source_sentence_uri,source_uri), abstract = answer['id']
		sentence = answer['context']
		# paragraph = None
		# if source_sentence_uri:
		# 	paragraph_iter = map(self.kg_manager.get_paragraph_text, self.kg_manager.source_dict[source_sentence_uri])
		# 	paragraph_iter = filter(lambda x: x, paragraph_iter)
		# 	paragraph_iter = filter(lambda x: len(x) <= self.kg_manager.max_paragraph_len, paragraph_iter)
		# 	paragraph_iter = list(paragraph_iter)
		# 	paragraph = max(paragraph_iter, key=len) if paragraph_iter else None

		# abstract = answer['doc']
		return {
			'abstract': abstract,
			'confidence': answer['similarity'],
			'syntactic_similarity': answer['syntactic_similarity'],
			'semantic_similarity': answer['semantic_similarity'],
			'extra_info': self.kg_manager.get_sub_graph(source_uri) if source_uri else None,
			# 'sentence': paragraph if paragraph else sentence, 
			'sentence': sentence, 
			'triple': triple, 
			'source_id': source_uri if source_uri else sentence, 
			'source_sentence_uri': source_sentence_uri,
		}

	def _init_sentence_classifier(self):
		self.logger.info('Initialising Sentence Classifier..')
		# Setup Sentence Classifier
		abstract_iter, context_iter, original_triple_iter, source_id_iter = zip(*filter(lambda x: x[0].strip() and x[1].strip(), self.kg_manager.get_sourced_graph()))
		id_doc_iter = tuple(zip(
			zip(original_triple_iter, source_id_iter), # id
			abstract_iter # doc
		))
		self._sentence_classifier.set_documents(id_doc_iter, tuple(context_iter))

	def _init_concept_classifier(self):
		self.logger.info('Initialising Concept Classifier..')
		self._concept_classifier.set_concept_description_dict(self.kg_manager.concept_description_dict)
		self.logger.info(f'This QA is now considering {len(self.kg_manager.aspect_uri_list)} concepts for question-answering.')
	
	def store_cache(self, cache_name):
		self._concept_classifier.store_cache(cache_name+'.concept_classifier.pkl')
		self._sentence_classifier.store_cache(cache_name+'.sentence_classifier.pkl')
		if self._sentence_summariser is not None:
			self._sentence_summariser.store_cache(cache_name+'.sentence_summariser.pkl')

	def load_cache(self, cache_name, save_if_init=True, **args):
		if self._sentence_classifier is None:
			self._sentence_classifier = SentenceClassifier(self.sentence_classifier_options)
			loaded_sentence_classifier = self._sentence_classifier.load_cache(cache_name+'.sentence_classifier.pkl')
			self._init_sentence_classifier()
			if not loaded_sentence_classifier and save_if_init:
				self._sentence_classifier.store_cache(cache_name+'.sentence_classifier.pkl')
		#######
		if self._concept_classifier is None:
			self._concept_classifier = ConceptClassifier(self.concept_classifier_options)
			loaded_concept_classifier = self._concept_classifier.load_cache(cache_name+'.concept_classifier.pkl')
			self._init_concept_classifier()
			if not loaded_concept_classifier and save_if_init:
				self._concept_classifier.store_cache(cache_name+'.concept_classifier.pkl')
		#######
		if self._sentence_summariser is not None:
			self._sentence_summariser.load_cache(cache_name+'.sentence_summariser.pkl')

	@staticmethod
	def get_question_answer_dict_quality(question_answer_dict, top=5):
		return {
			question: {
				# 'confidence': {
				# 	'best': answers[0]['confidence'],
				# 	'top_mean': sum(map(lambda x: x['confidence'], answers[:top]))/top,
				# },
				# 'syntactic_similarity': {
				# 	'best': answers[0]['syntactic_similarity'],
				# 	'top_mean': sum(map(lambda x: x['syntactic_similarity'], answers[:top]))/top,
				# },
				# 'semantic_similarity': {
				# 	'best': answers[0]['semantic_similarity'],
				# 	'top_mean': sum(map(lambda x: x['semantic_similarity'], answers[:top]))/top,
				# },
				'valid_answers_count': len(answers),
				'syntactic_similarity': answers[0]['syntactic_similarity'] if answers else 0,
				'semantic_similarity': answers[0]['semantic_similarity'] if answers else 0,
			}
			for question,answers in question_answer_dict.items()
		}

	@staticmethod
	def get_answer_question_pertinence_dict(question_answer_dict, update_answers=False):
		answer_question_pertinence_dict = {}
		for question,answers in question_answer_dict.items():
			for a in answers:
				question_pertinence_list = answer_question_pertinence_dict.get(a['sentence'],None)
				if question_pertinence_list is None:
					question_pertinence_list = answer_question_pertinence_dict[a['sentence']] = []
				question_pertinence_list.append(ArchetypePertinence(question, a['confidence']))
		if update_answers:
			for question,answers in question_answer_dict.items():
				for a in answers:
					a['question_pertinence_set'] = answer_question_pertinence_dict[a['sentence']]
		return answer_question_pertinence_dict

	@staticmethod
	def merge_duplicated_answers(question_answer_dict):
		# remove answers contained in other answers, replacing them with the longest answers
		valid_answers_list = flatten(question_answer_dict.values())
		valid_answer_sentence_list = list(map(lambda x: x['sentence'], valid_answers_list))
		for question,answers in question_answer_dict.items():
			for x in answers:
				x['sentence'] = max(filter(lambda y: x['sentence'] in y, valid_answer_sentence_list), key=len)
		for question in question_answer_dict.keys():
			question_answer_dict[question] = list(unique_everseen(question_answer_dict[question], key=lambda x: x['sentence']))
		return question_answer_dict

	@staticmethod
	def minimise_question_answer_dict(question_answer_dict):
		QuestionAnswererBase.logger.info('Minimising question answer dict')
		# remove duplicated answers
		answer_question_dict = QuestionAnswererBase.get_answer_question_pertinence_dict(question_answer_dict, update_answers=True)
		get_best_answer_archetype = lambda a: max(answer_question_dict[a['sentence']], key=lambda y: y.pertinence).archetype
		return {
			question: list(filter(lambda x: get_best_answer_archetype(x)==question, answers))
			for question,answers in question_answer_dict.items()
		}

	def sort_question_answer_dict(self, question_answer_dict, answer_to_question_max_similarity_threshold=None, answer_to_answer_max_similarity_threshold=None):
		for question, answer_list in question_answer_dict.items():
			answer_list = sorted(answer_list, key=lambda x: x['confidence'], reverse=True)
			answer_list = list(unique_everseen(answer_list, key=lambda x: (x['source_sentence_uri'],x['source_id'])))
			if answer_list and answer_to_question_max_similarity_threshold: # Answers contained in the question are not valid
				answer_list = self.sentence_classifier.filter_by_similarity_to_target(
					answer_list, 
					[question], 
					threshold=answer_to_question_max_similarity_threshold, 
					source_key=lambda a: a['sentence'], 
					target_key=lambda q: q
				)
			if answer_list and answer_to_answer_max_similarity_threshold: # Ignore similar-enough sentences with lower pertinence
				answer_list = self.sentence_classifier.remove_similar_labels(
					answer_list, 
					threshold=answer_to_answer_max_similarity_threshold, 
					key=lambda x: x['sentence'],
					without_context=True,
				)
			question_answer_dict[question] = answer_list
		return question_answer_dict

	@staticmethod
	def get_question_answer_overlap_dict(question_answer_dict):
		answer_question_dict = QuestionAnswererBase.get_answer_question_pertinence_dict(question_answer_dict)
		get_question_iter = lambda q,a_list: filter(lambda x: x!=q, (answer_question_pertinence_dict[a['sentence']].archetype for a in a_list))
		return {
			question: Counter(get_question_iter(question,answers))
			for question,answers in question_answer_dict.items()
		}

	def get_answer_relatedness_to_question(self, question_list, answer_list): 
		question_list = list(map(lambda x: x if question.endswith('?') else x+'?', question_list))
		return self.sentence_classifier.get_element_wise_similarity(question_list,answer_list, source_without_context=True, target_without_context=False)

################################################################################################################################################

	def summarise_question_answer_dict(self, question_answer_dict, ignore_non_grounded_answers=True, use_abstracts=False, summary_horizon=None, tree_arity=5, cut_factor=2, depth=None, similarity_threshold=0.3, remove_duplicates=True, min_size_for_summarising=None):
		# assert self.sentence_summariser is not None, "Missing sentence_summariser!"
		# print(json.dumps(question_answer_dict, indent=4))
		get_sentence = lambda x: x["abstract" if use_abstracts else 'sentence']
		if remove_duplicates:
			processed_sentence_set = set()
		question_summarised_answer_dict = {}
		for question, answer_list in question_answer_dict.items():
			answer_iter = iter(answer_list)
			if ignore_non_grounded_answers:
				answer_iter = filter(lambda x: x['extra_info'], answer_iter)
			if remove_duplicates:
				answer_iter = filter(lambda x: get_sentence(x) not in processed_sentence_set, answer_iter)
			answer_iter = unique_everseen(answer_iter, key=get_sentence)
			if summary_horizon:
				answer_list = tuple(itertools.islice(answer_iter, summary_horizon))
			else:
				answer_list = tuple(answer_iter)
			sentence_iter = map(get_sentence, answer_list)
			# sentence_iter = map(self.sentence_summariser.sentify, sentence_iter)
			sentence_list = list(sentence_iter)
			if use_abstracts:
				sentence_list = remove_similar_labels(sentence_list, similarity_threshold)
			integration_map = dict(zip(sentence_list,answer_list))
			summary_tree_list = self.sentence_summariser.summarise_sentence_list(sentence_list, tree_arity=tree_arity, cut_factor=cut_factor, depth=depth, min_size=min_size_for_summarising)
			self.sentence_summariser.integrate_summary_tree_list(integration_map, summary_tree_list)
			if summary_tree_list:
				if len(summary_tree_list) == 1:
					question_summarised_answer_dict[question] = summary_tree_list[0]
				else:
					# self.sentence_classifier.set_documents(tuple(enumerate(map(lambda x: x['summary'], summary_tree_list))))
					# # classify
					# classification_list = self.sentence_classifier.classify(query_list=[question], similarity_type='weighted', similarity_threshold=0, without_context=True)[0]
					# sorted_summary_tree_list = [
					# 	summary_tree_list[i]
					# 	for i in map(lambda x: x['id'], classification_list)
					# ]
					question_summarised_answer_dict[question] = {
						'summary': '',
						'children': summary_tree_list
					}
					# question_summarised_answer_dict[question] = {
					# 	'summary': summary_tree_list[0]['summary'],
					# 	'children': summary_tree_list[0]['children'] + summary_tree_list[1:]
					# }
			else:
				question_summarised_answer_dict[question] = {}
			###############################
			# for a in answer_list:
			# 	a['summary'] = a['sentence']
			# question_summarised_answer_dict[question] = {
			# 	'summary': self.sentence_summariser.summarise_sentence(' '.join(sentence_list))[0],
			# 	'children': answer_list
			# }
			if remove_duplicates:
				processed_sentence_set |= set(sentence_list)
		return question_summarised_answer_dict

	def annotate_question_summary_tree(self, question_summary_tree, similarity_threshold=0.8, max_concepts_per_alignment=1, tfidf_importance=None, is_preprocessed_content=False):
		if not question_summary_tree:
			return []
		def extract_sentence_list_from_tree(summary_tree):
			if not summary_tree:
				return []
			children = summary_tree.get('children',None)
			if not children:
				# label_list = [summary_tree['sentence']]
				# if 'summary' in summary_tree:
				# 	label_list.append(summary_tree['summary'])
				# return label_list
				return [summary_tree['sentence']] # extractive summaries contain the same words in the sentences
			label_list = []
			for c in children:
				label_list += extract_sentence_list_from_tree(c)
			return label_list
		sentence_list = flatten(map(lambda x: extract_sentence_list_from_tree(x), question_summary_tree.values()))
		sentence_list = tuple(unique_everseen(sentence_list))
		return self.concept_classifier.annotate(
			DocParser().set_content_list(sentence_list), 
			similarity_threshold=similarity_threshold, 
			max_concepts_per_alignment=max_concepts_per_alignment,
			tfidf_importance=tfidf_importance,
			concept_id_filter=lambda x: x in self.overview_aspect_set,
			is_preprocessed_content=is_preprocessed_content,
		)

	def annotate_taxonomical_view(self, taxonomical_view, similarity_threshold=0.8, max_concepts_per_alignment=1, tfidf_importance=None, is_preprocessed_content=False):
		if not taxonomical_view:
			return []
		sentence_iter = map(lambda y: y[-1], filter(lambda x: not is_url(x[-1]), taxonomical_view))
		return self.concept_classifier.annotate(
			DocParser().set_content_list(list(sentence_iter)), 
			similarity_threshold=similarity_threshold, 
			max_concepts_per_alignment=max_concepts_per_alignment,
			tfidf_importance=tfidf_importance,
			concept_id_filter=lambda x: x in self.overview_aspect_set,
			is_preprocessed_content=is_preprocessed_content,
		)

	def get_taxonomical_view(self, *arg, **args):
		return self.kg_manager.get_taxonomical_view(*arg, **args, concept_id_filter=lambda x: x in self.overview_aspect_set)
	