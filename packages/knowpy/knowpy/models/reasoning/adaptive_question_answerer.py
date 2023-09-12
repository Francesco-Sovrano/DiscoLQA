from knowpy.misc.doc_reader import DocParser
from knowpy.misc.cache_lib import create_cache, load_cache
from knowpy.models.reasoning.question_answerer import QuestionAnswerer
from knowpy.models.estimation.explainability_estimator import ExplainabilityEstimator

from knowpy.misc.utils import *
import itertools
import re
import json
import numpy as np

class AdaptiveQuestionAnswerer(QuestionAnswerer):
	def __init__(self, kg_manager, qa_dict_list, concept_classifier_options, sentence_classifier_options, dox_options=None, archetype_fitness_options=None, archetype_weight_dict=None, min_explainability=None, **args):
		super().__init__(kg_manager, concept_classifier_options, sentence_classifier_options, **args)
		##
		self.qa_dict_list = qa_dict_list #sorted(qa_dict_list, key=lambda x: (x['abstract'],x['sentence']))
		
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
		self._concept_qa_dict = None
		##
		self._aspect_explainability_dict = None
		self._min_explainability = min_explainability
		self._dox_options = dox_options
		self._archetype_fitness_options = archetype_fitness_options
		self._archetype_weight_dict = archetype_weight_dict

	def store_cache(self, cache_name):
		super().store_cache(cache_name)
		#######
		if self._aspect_explainability_dict is not None:
			create_cache(cache_name+'.aspect_WeDoX_dict.pkl', lambda: self._aspect_explainability_dict)

	def load_cache(self, cache_name, save_if_init=True, **args):
		super().load_cache(cache_name, save_if_init=save_if_init, **args)
		#######
		self._aspect_explainability_dict = load_cache(cache_name+'.aspect_WeDoX_dict.pkl')
		# self._aspect_explainability_dict = {k: v if not np.isnan(v) else 0. for k,v in self._aspect_explainability_dict.items()}
		# create_cache(cache_name+'.aspect_WeDoX_dict.pkl', lambda: self._aspect_explainability_dict)
		if self._aspect_explainability_dict is None and self._min_explainability and save_if_init:
			create_cache(cache_name+'.aspect_WeDoX_dict.pkl', lambda: self.aspect_explainability_dict) # do not use _aspect_explainability_dict with the initial underscore

	@property
	def aspect_explainability_dict(self):
		if not self._aspect_explainability_dict:
			self.logger.info(f'Building the explainability dict for {len(self.relevant_aspect_set)} aspects')
			
			self._aspect_explainability_dict = {}
			for aspect in self.tqdm(self.relevant_aspect_set):
				aspect_archetype_answers_dict = self.explainability_estimator.extract_archetypal_answers_per_aspect(
					aspect_uri_iter=(aspect,), 
					**self._dox_options
				)
				self._aspect_explainability_dict.update(
					self.explainability_estimator.get_aspect_WeDoX_dict(
						aspect_archetype_answers_dict, 
						self._dox_options, 
						self._archetype_fitness_options, 
						self._archetype_weight_dict
					)
				)
				# self.sentence_classifier.reset_cache()
			# we are not interested in abstracting
			# self._aspect_explainability_dict = self.explainability_estimator.get_abstract_WeDoX_dict(aspect_WeDoX_dict)
		return self._aspect_explainability_dict

	@property
	def overview_aspect_set(self):
		if self._overview_aspect_set is None:
			valid_concepts = filter(lambda x: self.kg_manager.is_relevant_aspect(x,ignore_leaves=True), self.concept_classifier.ids)
			if self._min_explainability:
				valid_concepts = filter(lambda x: self.aspect_explainability_dict.get(x,0) >= self._min_explainability, valid_concepts)
			self._overview_aspect_set = set(valid_concepts)
			# Betweenness centrality quantifies the number of times a node acts as a bridge along the shortest path between two other nodes.
			if self.betweenness_centrality is not None:
				filtered_betweenness_centrality = dict(filter(lambda x: x[-1] > 0, self.betweenness_centrality.items()))
				self._overview_aspect_set &= filtered_betweenness_centrality.keys()
		return self._overview_aspect_set

	def get_most_interesting_concepts(self, top_k=5, text=None, similarity_threshold=None, max_concepts_per_alignment=1, tfidf_importance=None, is_preprocessed_content=True):
		sorted_aspect_explainability_dict = sorted(self.aspect_explainability_dict.items(), key=lambda x: x[-1], reverse=True)
		most_interesting_concept_iter = (c for c,_ in sorted_aspect_explainability_dict)
		if text:
			annotation_list = self.concept_classifier.annotate(
				DocParser().set_content_list([text]), 
				similarity_threshold= similarity_threshold, 
				max_concepts_per_alignment= max_concepts_per_alignment,
				tfidf_importance= tfidf_importance,
				concept_id_filter= lambda x: x in self.overview_aspect_set,
				is_preprocessed_content= is_preprocessed_content,
			)
			relevant_concept_set = set((a['annotation'] for a in annotation_list))
			most_interesting_concept_iter = filter(lambda x: x in relevant_concept_set, most_interesting_concept_iter)
		return tuple(itertools.islice(most_interesting_concept_iter, top_k))

	def get_most_interesting_query_list(self, concept_uri, is_new_content_fn=None, top_k=5, adaptivity_options=None, max_query_size=None):
		dox_options = dict(self._dox_options)
		if adaptivity_options:
			dox_options['adaptivity_options'] = adaptivity_options
		
		# Get possible Q/A about concept_uri
		self.logger.info('AdaptiveQuestionAnswerer: Get possible Q/A about concept_uri')
		if concept_uri not in self.kg_manager.label_dict:
			self.logger.info(f'AdaptiveQuestionAnswerer: No label found for {concept_uri}')
			return []
		label_list = [
			re.compile(f'(^{x} | {x} | {x}$)', re.IGNORECASE)
			for x in map(re.escape, self.kg_manager.label_dict[concept_uri])
		]
		qa_dict_iter = (
			qa_dict
			for qa_dict in self.qa_dict_list
			if next(filter(lambda x: re.search(x, qa_dict['question']), label_list), None)
		)
		if max_query_size:
			qa_dict_iter = filter(lambda x: len(x['question']) <= max_query_size, qa_dict_iter)
		qa_dict_list = list(qa_dict_iter)
		if not qa_dict_list:
			return []
		# Compute DoX of filtered answers on concept_uri
		self.logger.info('AdaptiveQuestionAnswerer: Compute DoX of filtered answers on concept_uri')
		if self._archetype_fitness_options is None:
			self._archetype_fitness_options = {}
		answer_aspect_dox_dict = self.explainability_estimator.get_sentence_aspect_dox_dict(
			aspect_uri_list=[concept_uri],
			overview_options=dox_options,
			archetype_fitness_options=self._archetype_fitness_options,
			# is_new_content_fn=is_new_content_fn,
		)
		answer_dox_dict = {
			sentence: self.explainability_estimator.get_weighted_degree_of_explainability(
				aspect_dict[concept_uri], 
				self._archetype_weight_dict
			)
			for sentence, aspect_dict in answer_aspect_dox_dict.items()
		}
		# Order questions by decreasing DoX
		self.logger.info('AdaptiveQuestionAnswerer: Order questions by decreasing DoX')
		qa_dict_list = sorted(
			qa_dict_list, 
			key=lambda x: answer_dox_dict.get(x['sentence'], -1), 
			reverse=True
		)
		# Get top_k questions
		# return qa_dict_list[:top_k]
		return [x['question'] for x in qa_dict_list[:top_k]]

	@staticmethod
	def adapt_qa_dict(question_answer_dict, is_new_content_fn=None, old_content_confidence_scaling_factor=0.5, answer_pertinence_threshold=0.55, **args):
		if is_new_content_fn is None or old_content_confidence_scaling_factor == 1:
			return question_answer_dict
		AdaptiveQuestionAnswerer.logger.info('Adapting QA dict')
		for question, answer_dict_list in question_answer_dict.items():
			for answer_dict in answer_dict_list:
				if not is_new_content_fn(answer_dict['sentence']):
					answer_dict['confidence'] *= old_content_confidence_scaling_factor
		# Filter and sort answers by confidence
		for question, answer_dict_list in question_answer_dict.items():
			answer_dict_iter = answer_dict_list
			if answer_pertinence_threshold is not None:
				answer_dict_iter = filter(lambda x: x['confidence'] >= answer_pertinence_threshold, answer_dict_iter)
			answer_dict_list = sorted(answer_dict_iter, key=lambda x: x['confidence'], reverse=True)
			question_answer_dict[question] = answer_dict_list
		return question_answer_dict

	def ask(self, question_list, adaptivity_options=None, **args):
		result = super().ask(question_list, **args)
		if not adaptivity_options:
			return result
		return AdaptiveQuestionAnswerer.adapt_qa_dict(result, **adaptivity_options, **args)
		
	def get_concept_overview(self, concept_uri=None, query_template_list_generator_options=None, adaptivity_options=None, **args):
		if args.get('query_template_list', None) is not None:
			result = super().get_concept_overview(concept_uri=concept_uri, **args)
			if not adaptivity_options:
				return result
			return AdaptiveQuestionAnswerer.adapt_qa_dict(result, **adaptivity_options, **args)
		assert concept_uri, f"{concept_uri} is not a valid concept_uri"
		assert query_template_list_generator_options, f"not a valid query_template_list_generator_options having no query_template_list"
		
		query_template_list = self.get_most_interesting_query_list(concept_uri, **query_template_list_generator_options)
		if not query_template_list:
			return {}
		result = super().get_concept_overview(query_template_list=query_template_list, concept_uri=concept_uri, **args)
		if not adaptivity_options:
			return result
		if not result:
			return {}
		return AdaptiveQuestionAnswerer.adapt_qa_dict(result, **adaptivity_options, **args)
