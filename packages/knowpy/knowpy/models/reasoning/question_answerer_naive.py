import os
import numpy as np
from more_itertools import unique_everseen
import sentence_transformers as st
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import logging

from knowpy.misc.doc_reader import DocParser
from knowpy.models.reasoning.question_answerer import QuestionAnswerer
from knowpy.models.knowledge_extraction.couple_extractor import filter_invalid_sentences
# from knowpy.models.knowledge_extraction.question_answer_extractor import QuestionAnswerExtractor

from knowpy.misc.jsonld_lib import *
from knowpy.misc.utils import *
from knowpy.misc.cache_lib import create_cache, load_cache
from knowpy.models.reasoning import is_not_wh_word
import hnswlib
from knowpy.models.estimation.explainability_estimator import ExplainabilityEstimator

get_hybrid_score = lambda x: x['confidence']*x['score']

class QuestionAnswererNaive(QuestionAnswerer):
	def __init__(self, kg_manager, qa_dict_list, concept_classifier_options, sentence_classifier_options, answer_summariser_options=None, betweenness_centrality=None, fast_knn_search=None):
		super().__init__(kg_manager, concept_classifier_options, sentence_classifier_options, answer_summariser_options, betweenness_centrality)
		##
		self.qa_dict_list = sorted(qa_dict_list, key=lambda x: (x['abstract'],x['sentence']))
		self.fast_knn_search = fast_knn_search
		
		self.explainability_estimator = ExplainabilityEstimator(self)
		# Build content-to-source dict
		self.content_to_source_dict = {}
		for source_uri,sentence_list in self.kg_manager.content_dict.items():
			for sentence in sentence_list:
				source_uri_list = self.content_to_source_dict.get(sentence,None)
				if source_uri_list is None:
					source_uri_list = self.content_to_source_dict[sentence] = []
				source_uri_list.append(source_uri)
		##
		self._knn_index = None
		self._corpus_embeddings = None
		self._concept_qa_dict = None
		
	def store_cache(self, cache_name):
		self.logger.info(f'QuestionAnswererNaive: Storing cache')
		super().store_cache(cache_name)

		if self._knn_index is not None:
			self._knn_index.save_index(cache_name+'.hnswlib.index')

	def load_cache(self, cache_name, init=True, save_if_init=True, **args):
		self.logger.info(f'QuestionAnswererNaive: Loading cache')
		super().load_cache(cache_name, init=init, save_if_init=save_if_init)

		if os.path.exists(cache_name+'.hnswlib.index'):
			dim = self.sentence_classifier.sentence_embedding_fn([('abstract','context')], without_context=False)[0].shape[-1]
			self._knn_index = hnswlib.Index(space='ip', dim=dim)
			self._knn_index.load_index(cache_name+'.hnswlib.index')
			self._knn_index.set_ef(self.fast_knn_search.get('ef', 64)) # ef should always be > top_k_hits # Note that the parameter is currently not saved along with the index, so you need to set it manually after loading.
		elif init:
			if self.initialise_fast_semantic_search() and save_if_init:
				self._knn_index.save_index(cache_name+'.hnswlib.index')

	def _init_sentence_classifier(self):
		if not self.fast_knn_search:
			self.logger.info(f'QuestionAnswererNaive: Init QADisco without fast_knn_search')
			question_answer_list = [
				(x['abstract'], x['sentence'])
				for x in self.qa_dict_list
				if x['sentence'].strip() and x['abstract'].strip()
			]
			self.sentence_classifier.sentence_embedding_fn(question_answer_list, without_context=False)

	def initialise_fast_semantic_search(self):
		if self._knn_index is not None:
			return False
		if not self.fast_knn_search:
			return False
		self.logger.info(f'QuestionAnswererNaive: Init fast_semantic_search')
		#We use Inner Product (dot-product) as Index. We will normalize our vectors to unit length, then is Inner Product equal to cosine similarity
		self._knn_index = hnswlib.Index(space='ip', dim=self.corpus_embeddings[0].shape[-1])
		### Create the HNSWLIB index
		self.logger.info(f"Start creating HNSWLIB index with {self.fast_knn_search['ef']} k_trees")
		self._knn_index.init_index(max_elements=len(self.corpus_embeddings), ef_construction=self.fast_knn_search.get('ef_construction',400), M=self.fast_knn_search.get('M',64))
		# Then we train the index to find a suitable clustering
		self._knn_index.add_items(self.corpus_embeddings, list(range(len(self.corpus_embeddings))))
		self._knn_index.set_ef(self.fast_knn_search.get('ef', 64)) # ef should always be > top_k_hits # Note that the parameter is currently not saved along with the index, so you need to set it manually after loading.
		self.corpus_embeddings = None # save memory, we don't need corpus_embeddings anymore
		return True

	@property
	def corpus_embeddings(self):
		if self._corpus_embeddings is None:
			self.logger.info(f'QuestionAnswererNaive: Building corpus_embeddings')
			# self.logger.info('Embedding QA matrix..')
			question_answer_list = list(map(lambda x:(x['abstract'],x['sentence']),self.qa_dict_list))
			self._corpus_embeddings = self.sentence_classifier.sentence_embedding_fn(question_answer_list, without_context=False)
			assert len(self._corpus_embeddings)==len(self.qa_dict_list), f"corpus_embeddings ({len(self._corpus_embeddings)}) should have the same length of qa_dict_list ({len(self.qa_dict_list)})"
		return self._corpus_embeddings

	@corpus_embeddings.setter
	def corpus_embeddings(self, value):
		self._corpus_embeddings = value

	@property
	def knn_index(self):
		self.initialise_fast_semantic_search()
		return self._knn_index

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
				# 'syntactic_similarity': answers[0]['syntactic_similarity'],
				'semantic_similarity': answers[0]['semantic_similarity'],
			}
			for question,answers in question_answer_dict.items()
			if answers
		}

	def ask(self, query_list, query_concept_similarity_threshold=0.55, answer_pertinence_threshold=None, with_numbers=True, remove_stopwords=False, lemmatized=False, keep_the_n_most_similar_concepts=1, include_super_concepts_graph=True, include_sub_concepts_graph=True, concept_label_filter=is_not_wh_word, answer_to_question_max_similarity_threshold=0.9502, answer_to_answer_max_similarity_threshold=0.9502, query_known_concepts=None, only_relevant_concepts=False, top_k=None, **args):
		# Searching for answers..
		query_embedding_list = self.sentence_classifier.sentence_embedding_fn(query_list, without_context=True)
		top_k = min(top_k, len(self.qa_dict_list)) if top_k else len(self.qa_dict_list)
		if self.fast_knn_search:
			#We use hnswlib knn_query method to find the top_k_hits
			semantic_search_results = []
			for query_embedding in query_embedding_list:
				corpus_ids, distances = self.knn_index.knn_query(query_embedding, k=top_k)
				# We extract corpus ids and scores for the first query
				hits = [{'corpus_id': int(cid)-1, 'score': 1-score} for cid, score in zip(corpus_ids[0], distances[0])]
				hits = sorted(hits, key=lambda x: x['score'], reverse=True)
				semantic_search_results.append(hits)
		else:
			semantic_search_results = st.util.semantic_search(
				query_embeddings= query_embedding_list,
				corpus_embeddings= self.corpus_embeddings,
				top_k= top_k,
				score_function= st.util.dot_score,
			)
		# Organising answers by relevant concepts..
		question_answer_dict = {}
		for i,(query,query_results) in enumerate(zip(query_list,semantic_search_results)):
			query_embedding = query_embedding_list[i]
			if only_relevant_concepts:
				if query_known_concepts is None or query not in query_known_concepts:
					concepts_dict = self.concept_classifier.get_concept_dict(
						DocParser().set_content_list([query]),
						similarity_threshold=query_concept_similarity_threshold, 
						with_numbers=with_numbers, 
						remove_stopwords=remove_stopwords, 
						lemmatized=lemmatized,
						concept_label_filter=concept_label_filter,
					)
					concept_set = set(unique_everseen((
						concept_similarity_dict["id"]
						for concept_label, concept_count_dict in concepts_dict.items()
						for concept_similarity_dict in itertools.islice(
							unique_everseen(concept_count_dict["similar_to"], key=lambda x: x["id"]), 
							keep_the_n_most_similar_concepts
						)
					)))
				else:
					concept_set = set(query_known_concepts[query])
				expanded_concept_set = set(concept_set)
				# Get sub-classes
				if include_sub_concepts_graph:
					sub_concept_set = self.kg_manager.adjacency_list.get_predicate_chain(
						concept_set = concept_set, 
						predicate_filter_fn = lambda x: x == SUBCLASSOF_PREDICATE, 
						direction_set = ['in'],
						depth = None,
					)
					expanded_concept_set |= sub_concept_set
				# Get super-classes
				if include_super_concepts_graph:
					super_concept_set = self.kg_manager.adjacency_list.get_predicate_chain(
						concept_set = concept_set, 
						predicate_filter_fn = lambda x: x == SUBCLASSOF_PREDICATE, 
						direction_set = ['out'],
						depth = None,
					)
					expanded_concept_set |= super_concept_set
				self.logger.info(f'Concepts in "{query}": {expanded_concept_set}')

				relevant_concept_label_list = sorted(
					flatten((
						[
							(label.lower(),uri)
							for label in self.kg_manager.label_dict[uri]
						]
						for uri in expanded_concept_set 
						if uri in self.kg_manager.label_dict
					)),
					key=lambda x: len(x),
					reverse=True,
				)
				self.logger.info(f'Relevant labels found in query "{query}": {relevant_concept_label_list}')

				def get_relevant_uri_tuple(txt):
					txt = txt.lower()
					return tuple(sorted(unique_everseen([
						uri
						for label,uri in relevant_concept_label_list
						if label in txt
					])))

			# uri_answer_dict = {}
			answer_dict_list = []
			sentence_set = set()
			for result_dict in query_results:
				#print(result_dict)
				hit_embedding = self.corpus_embeddings[result_dict['corpus_id']] if not self.fast_knn_search else self.knn_index.get_items([result_dict['corpus_id']])[0]
				confidence = float(cosine_similarity([query_embedding], [hit_embedding])[0][0])
				# confidence = float(result_dict['score'])
				if answer_pertinence_threshold is not None and confidence < answer_pertinence_threshold:
					continue
				qa_dict = self.qa_dict_list[result_dict['corpus_id']]
				sentence = qa_dict['sentence']
				if sentence in sentence_set: # ignore duplicates
					continue
				sentence_set.add(sentence)
				abstract = qa_dict['abstract']
				if only_relevant_concepts:
					relevant_uri_tuple = get_relevant_uri_tuple(abstract)
					if not relevant_uri_tuple: # ignore elements that do not contain relevant labels
						continue
				related_question = qa_dict['question']
				related_answer = qa_dict['answer']
				related_answer_type = qa_dict['type']
				source_uri_list = self.content_to_source_dict.get(sentence,[])
				if not source_uri_list:
					self.logger.debug(f'Error: sentence not found: {sentence}')
				answer_dict_list += [
					{
						'abstract': abstract,
						'confidence': confidence,
						'score': float(result_dict['score']),
						# 'relevant_to': relevant_uri_tuple,
						'syntactic_similarity': confidence,
						'semantic_similarity': confidence,
						'extra_info': self.kg_manager.get_sub_graph(source_uri) if source_uri else None,
						'sentence': sentence, 
						'triple': (related_question,related_answer_type,related_answer), 
						'source_id': source_uri if source_uri else sentence, 
					}
					for source_uri in source_uri_list
				]
			# answers contained in the question are not valid
			if answer_to_question_max_similarity_threshold is not None and answer_dict_list:
				answer_dict_list = self.sentence_classifier.filter_by_similarity_to_target(
					answer_dict_list, 
					[query], 
					threshold=answer_to_question_max_similarity_threshold, 
					source_key=lambda a: a['abstract'], 
					target_key=lambda q: q
				)
			# ignore similar-enough sentences with lower pertinence
			if answer_to_answer_max_similarity_threshold is not None and answer_dict_list:
				answer_dict_list = self.sentence_classifier.remove_similar_labels(
					answer_dict_list, 
					threshold=answer_to_answer_max_similarity_threshold, 
					key=lambda x: (x['abstract'],x['sentence']),
					without_context=False,
				)
			answer_dict_list = sorted(answer_dict_list, key=get_hybrid_score, reverse=True)
			# question_answer_dict[query] = remove_similar_labels(
			# 	list(answer_dict_list), 
			# 	key=lambda x: x['abstract']
			# )
			question_answer_dict[query] = answer_dict_list
		return question_answer_dict

	def get_concept_overview(self, query_template_list=None, concept_uri=None, concept_label=None, sort_archetypes_by_relevance=True, minimise=True, only_valid_sentences=True, **args):
		assert concept_uri, f"{concept_uri} is not a valid concept_uri"
		if query_template_list is None:
			query_template_list = list(self.archetypal_questions_dict.values())
		if not concept_label:
			concept_label = self.kg_manager.get_label(concept_uri)
		# set consider_incoming_relations to False with concept-centred generic questions (e.g. what is it?), otherwise the answers won't be the sought ones
		query_list = tuple(map(lambda x:x.replace('{X}',concept_label), query_template_list))
		query_known_concepts = {
			query: [concept_uri]
			for query in query_list
		}
		answer_to_question_max_similarity_threshold = args.get('answer_to_question_max_similarity_threshold',None)
		args['answer_to_question_max_similarity_threshold'] = None
		question_answer_dict = self.ask(query_list, query_known_concepts=query_known_concepts, **args)
		question_answer_values = question_answer_dict.values()
		# answers contained in the question are not valid
		if answer_to_question_max_similarity_threshold is not None:
			question_answer_values = (
				self.sentence_classifier.filter_by_similarity_to_target(
					answer_list, 
					[concept_label], 
					threshold=answer_to_question_max_similarity_threshold, 
					source_key=lambda a: a['abstract'], 
					target_key=lambda q: q,
				)
				for answer_list in question_answer_values
			)
		# filter invalid sentences not being useful for any overview
		if only_valid_sentences:
			question_answer_values = (
				filter_invalid_sentences(self, answer_list, key=lambda x: x['sentence'], avoid_coreferencing=False)
				for answer_list in question_answer_values
			)
		question_answer_items = zip(query_template_list,question_answer_values)
		question_answer_items = filter(lambda x: x[-1], question_answer_items) # remove unanswered questions
		if sort_archetypes_by_relevance:
			question_answer_items = sorted(question_answer_items, key=lambda x: get_hybrid_score(x[-1][0]), reverse=True)
		question_answer_dict = dict(question_answer_items)
		if minimise:
			question_answer_dict = self.minimise_question_answer_dict(question_answer_dict)
		return question_answer_dict

	######################################################################
	######################################################################

	@property
	def concept_qa_dict(self):
		if self._concept_qa_dict is None:
			self._concept_qa_dict = {
				uri: [
					qa_dict
					for qa_dict in self.qa_dict_list
					if next(filter(lambda l: l.lower() in qa_dict['abstract'].lower(), label_list), None)
				]
				for uri, label_list in self.kg_manager.label_dict.items()
			}
		return self._concept_qa_dict

	def get_adaptive_overview(self, concept_uri, old_answer_sentence_set=None, top_k=5, overview_options=None, archetype_fitness_options=None, archetype_weight_dict=None):
		if old_answers is None:
			old_answers = []
		if archetype_fitness_options is None:
			archetype_fitness_options = {}
		# Get possible questions about concept_uri
		# Get answers to these questions
		qa_dict_list = self.concept_qa_dict[concept_uri]
		# Filter previously given answers
		# qa_dict_list = list(filter(lambda x: x['sentence'] not in old_answer_sentence_set, qa_dict_list))
		# Compute DoX of filtered answers on concept_uri
		answer_aspect_dox_dict = self.explainability_estimator.get_sentence_aspect_dox_dict(
			aspect_uri_list=[concept_uri],
			overview_options=overview_options,
			archetype_fitness_options=archetype_fitness_options
		)
		answer_dox_dict = {
			sentence: self.explainability_estimator.get_weighted_degree_of_explainability(
				aspect_dict[concept_uri], 
				archetype_weight_dict
			)
			for sentence, aspect_dict in answer_aspect_dox_dict.items()
		}
		# Order questions by decreasing DoX
		qa_dict_list = sorted(qa_dict_list, key=lambda x: answer_dox_dict[x['sentence']], reverse=True)
		# Return the top_k hits
		qa_dict_list = qa_dict_list[:top_k]
		# Build question-answer dict
		# source_uri_list = self.content_to_source_dict.get(x['sentence'],[])
		question_answer_dict = {
			x['question']: [
				{
					'abstract': x['abstract'],
					# 'confidence': confidence,
					'score': answer_dox_dict[x['sentence']],
					# 'syntactic_similarity': confidence,
					# 'semantic_similarity': confidence,
					# 'extra_info': self.kg_manager.get_sub_graph(source_uri) if source_uri else None,
					'sentence': x['sentence'], 
					'triple': (x['question'], x['type'], x['answer']), 
					# 'source_id': source_uri if source_uri else sentence, 
				}
				# for source_uri in source_uri_list
			]
			for x in qa_dict_list
		}
		return question_answer_dict
