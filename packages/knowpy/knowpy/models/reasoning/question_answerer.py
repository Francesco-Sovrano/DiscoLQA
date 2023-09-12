import json
from more_itertools import unique_everseen
import itertools
import logging

from knowpy.misc.doc_reader import DocParser

from knowpy.misc.jsonld_lib import *
from knowpy.models.reasoning.question_answerer_base import *
from knowpy.models.reasoning import is_not_wh_word
from knowpy.models.knowledge_extraction.couple_extractor import filter_invalid_sentences

class QuestionAnswerer(QuestionAnswererBase):

	def _init_sentence_classifier(self):
		pass

	def find_answers_in_concept_graph(self, query_list, concept_uri, question_answer_dict, answer_pertinence_threshold=0.55, add_external_definitions=False, include_super_concepts_graph=False, include_sub_concepts_graph=False, consider_incoming_relations=False, tfidf_importance=None, answer_to_question_max_similarity_threshold=0.97, answer_to_answer_max_similarity_threshold=0.97, use_weak_pointers=False):
		self.logger.info(f"Getting aspect_graph for {concept_uri}")
		concept_graph = self.kg_manager.get_aspect_graph(
			concept_uri=concept_uri, 
			add_external_definitions=add_external_definitions, 
			include_super_concepts_graph=include_super_concepts_graph, 
			include_sub_concepts_graph=include_sub_concepts_graph, 
			consider_incoming_relations=consider_incoming_relations,
			filter_fn=lambda x: '{obj}' in x[1],
		)
		self.logger.debug('######## Concept Graph ########')
		self.logger.debug(f"{concept_uri} has {len(concept_graph)} triplets")
		self.logger.debug(json.dumps(concept_graph, indent=4))
		# Extract sourced triples
		self.logger.info(f"Getting get_sourced_graph_from_aspect_graph for {concept_uri}")
		sourced_natural_language_triples_set = self.kg_manager.get_sourced_graph_from_aspect_graph(concept_graph)
		if len(sourced_natural_language_triples_set) <= 0:
			self.logger.warning(f'Missing: {concept_uri}')
			return
		# sourced_natural_language_triples_set.sort(key=str) # only for better summary caching
		# Setup Sentence Classifier
		abstract_iter, context_iter, original_triple_iter, source_id_iter = zip(*sourced_natural_language_triples_set)
		id_doc_iter = tuple(zip(
			zip(original_triple_iter, source_id_iter, abstract_iter), # id
			map(lambda x: x.split('?')[-1] if '?' in x else x, abstract_iter) if use_weak_pointers else abstract_iter # doc
		))
		self.sentence_classifier.set_documents(id_doc_iter, tuple(context_iter))
		# classify
		classification_dict_gen = self.sentence_classifier.classify(
			query_list=query_list, 
			similarity_type='weighted', 
			similarity_threshold=answer_pertinence_threshold, 
			without_context=True, 
			tfidf_importance=tfidf_importance
		)
		# Add missing questions to question_answer_dict
		for question in query_list:
			if question not in question_answer_dict:
				question_answer_dict[question] = []
		# Format Answers
		for i,(question, answer_iter) in enumerate(zip(query_list, classification_dict_gen)):
			answer_iter = filter(lambda x: x['doc'], answer_iter)
			question_answer_dict[question] += map(self.get_formatted_answer, answer_iter)
		return question_answer_dict

################################################################################################################################################

	def ask(self, question_list, query_concept_similarity_threshold=0.55, answer_pertinence_threshold=0.55, with_numbers=True, remove_stopwords=False, lemmatized=False, keep_the_n_most_similar_concepts=1, add_external_definitions=False, include_super_concepts_graph=True, include_sub_concepts_graph=True, consider_incoming_relations=True, tfidf_importance=None, concept_label_filter=is_not_wh_word, answer_to_question_max_similarity_threshold=0.97, answer_to_answer_max_similarity_threshold=0.97, use_weak_pointers=False, filter_fn=None, top_k=None, minimise=False, **args):
		# set consider_incoming_relations to False with concept-centred generic questions (e.g. what is it?), otherwise the answers won't be the sought ones
		self.logger.info(f'Extracting concepts from question_list: {json.dumps(question_list, indent=4)}..')
		concepts_dict = self.concept_classifier.get_concept_dict(
			doc_parser=DocParser().set_content_list(question_list),
			similarity_threshold=query_concept_similarity_threshold, 
			with_numbers=with_numbers, 
			remove_stopwords=remove_stopwords, 
			lemmatized=lemmatized,
			concept_label_filter=concept_label_filter,
			size=keep_the_n_most_similar_concepts,
		)
		self.logger.debug('######## Concepts Dict ########')
		self.logger.debug(json.dumps(concepts_dict, indent=4))
		# Group queries by concept_uri
		concept_uri_query_dict = {}
		# print(json.dumps(concepts_dict, indent=4))
		for concept_label, concept_count_dict in concepts_dict.items():
			for concept_similarity_dict in itertools.islice(unique_everseen(concept_count_dict["similar_to"], key=lambda x: x["id"]), max(1,keep_the_n_most_similar_concepts)):
				concept_uri = concept_similarity_dict["id"]
				concept_query_set = concept_uri_query_dict.get(concept_uri,None)
				if concept_query_set is None:
					concept_query_set = concept_uri_query_dict[concept_uri] = set()
				concept_query_set.update((
					sent_dict["paragraph_text"]
					for sent_dict in concept_count_dict["source_list"]
				))
		# For every aligned concept, extract from the ontology all the incoming and outgoing triples, thus building a partial graph (a view).
		question_answer_dict = {}
		for concept_uri, concept_query_set in concept_uri_query_dict.items():
			self.logger.info(f'Extracting answers related to {concept_uri}..')
			self.find_answers_in_concept_graph(
				query_list= list(concept_query_set), 
				concept_uri= concept_uri, 
				question_answer_dict= question_answer_dict, 
				answer_pertinence_threshold= answer_pertinence_threshold,
				add_external_definitions= add_external_definitions,
				include_super_concepts_graph= include_super_concepts_graph, 
				include_sub_concepts_graph= include_sub_concepts_graph, 
				consider_incoming_relations= consider_incoming_relations,
				tfidf_importance= tfidf_importance,
				use_weak_pointers= use_weak_pointers,
			)
		####################################
		## Sort and filter duplicated answers
		self.logger.info(f'Sorting answers..')
		question_answer_dict = self.sort_question_answer_dict(question_answer_dict, answer_to_question_max_similarity_threshold=answer_to_question_max_similarity_threshold, answer_to_answer_max_similarity_threshold=answer_to_answer_max_similarity_threshold)
		####################################
		question_answer_items = zip(question_list, question_answer_dict.values())
		if filter_fn: # remove unwanted answers
			question_answer_items = map(lambda x: (x[0], list(filter(filter_fn,x[-1]))), question_answer_items) # remove unwanted answers
		question_answer_dict = dict(question_answer_items)
		if minimise:
			question_answer_dict = self.merge_duplicated_answers(question_answer_dict)
		return question_answer_dict

	def get_concept_overview(self, query_template_list=None, concept_uri=None, concept_label=None, answer_pertinence_threshold=0.3, add_external_definitions=True, include_super_concepts_graph=True, include_sub_concepts_graph=True, consider_incoming_relations=True, tfidf_importance=None, sort_archetypes_by_relevance=True, answer_to_question_max_similarity_threshold=0.97, answer_to_answer_max_similarity_threshold=0.97, minimise=True, use_weak_pointers=False, **args):
		assert concept_uri, f"{concept_uri} is not a valid concept_uri"
		if query_template_list is None:
			query_template_list = list(QuestionAnswererBase.archetypal_questions_dict.values())
		elif not query_template_list:
			return {}
		# set consider_incoming_relations to False with concept-centred generic questions (e.g. what is it?), otherwise the answers won't be the sought ones
		if not concept_label:
			concept_label = self.kg_manager.get_label(concept_uri)
		question_answer_dict = {}
		self.logger.info(f'get_concept_overview {concept_uri}: finding answers in concept graph..')
		self.find_answers_in_concept_graph(
			query_list= tuple(map(lambda x:x.replace('{X}',concept_label), query_template_list)), 
			concept_uri= concept_uri, 
			question_answer_dict= question_answer_dict, 
			answer_pertinence_threshold= answer_pertinence_threshold,
			add_external_definitions= add_external_definitions,
			include_super_concepts_graph= include_super_concepts_graph, 
			include_sub_concepts_graph= include_sub_concepts_graph, 
			consider_incoming_relations= consider_incoming_relations,
			tfidf_importance= tfidf_importance,
			use_weak_pointers = use_weak_pointers,
		)
		#######################
		self.logger.info(f'Sorting answers..')
		question_answer_dict = self.sort_question_answer_dict(question_answer_dict, answer_to_question_max_similarity_threshold=answer_to_question_max_similarity_threshold, answer_to_answer_max_similarity_threshold=answer_to_answer_max_similarity_threshold)
		#######################
		question_answer_values = question_answer_dict.values()
		# question_answer_items = question_answer_dict.items()
		question_answer_items = zip(query_template_list, question_answer_values)
		question_answer_items = filter(lambda x: x[-1], question_answer_items) # remove unanswered questions
		# re_exp = re.compile(f' *{re.escape(concept_label)}')
		# question_answer_items = map(lambda x: (re.sub(re_exp,'',x[0]), x[1]), question_answer_items)
		question_answer_dict = dict(question_answer_items)
		if minimise:
			question_answer_dict = self.minimise_question_answer_dict(question_answer_dict)
		if sort_archetypes_by_relevance:
			question_answer_dict = dict(sorted(question_answer_dict.items(), key=lambda x: sum(map(lambda y: y['confidence'], x[-1])), reverse=True))
			# question_answer_dict = dict(sorted(question_answer_dict.items(), key=lambda x: x[-1][0]['confidence'] if x[-1] else float('-inf'), reverse=True))
		return question_answer_dict

