import sched, time
import json
import math
import numpy as np
from os import mkdir, path as os_path
base_path = os_path.dirname(os_path.abspath(__file__))
cache_path = os_path.join(base_path,'cache')
document_path = os_path.join(base_path,'documents')

# from knowpy.models.knowledge_extraction.ontology_builder import OntologyBuilder
from knowpy.models.knowledge_extraction.knowledge_graph_builder import KnowledgeGraphBuilder
from knowpy.models.reasoning.question_answerer import QuestionAnswerer
from knowpy.models.reasoning.question_answerer_naive import QuestionAnswererNaive
from knowpy.models.reasoning.adaptive_question_answerer import AdaptiveQuestionAnswerer
from knowpy.models.knowledge_extraction.knowledge_graph_manager import KnowledgeGraphManager
from knowpy.models.knowledge_extraction.couple_extractor import filter_invalid_sentences

from knowpy.models.knowledge_extraction.question_answer_extractor import QuestionAnswerExtractor
from knowpy.misc.doc_reader import load_or_create_cache, DocParser, get_document_list
from knowpy.misc.graph_builder import get_concept_description_dict, get_betweenness_centrality, save_graphml
from knowpy.misc.levenshtein_lib import labels_are_contained, remove_similar_labels
from knowpy.misc.utils import *
from more_itertools import unique_everseen
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool

import sys
import logging
logger = logging.getLogger('knowpy')
logger.setLevel(logging.INFO)
# logger.setLevel(logging.ERROR)
logger.addHandler(logging.StreamHandler(sys.stdout))

logger = logging.getLogger('quansx')
logger.setLevel(logging.INFO)
# logger.setLevel(logging.ERROR)
logger.addHandler(logging.StreamHandler(sys.stdout))

# import sys
# _,model_type,info_type = sys.argv

# model_type = model_type.casefold()
# info_type = info_type.casefold()
WITH_ANNOTATIONS = False
GRAPH_CLEANING_OPTIONS = {
	'remove_stopwords': False,
	'remove_numbers': False,
	'avoid_jumps': True,
	'parallel_extraction': True,
}

OVERVIEW_OPTIONS = {
	'answer_horizon': 10,
	######################
	## QuestionAnswerer stuff
	'tfidf_importance': 0,
	'answer_pertinence_threshold': 0.3, 
	'answer_to_question_max_similarity_threshold': None,
	'answer_to_answer_max_similarity_threshold': 0.98,
	'use_weak_pointers': True,
	'top_k': 100,
	######################
	'keep_the_n_most_similar_concepts': 1, 
	'query_concept_similarity_threshold': 0.55, 
	'include_super_concepts_graph': False, 
	'include_sub_concepts_graph': True, 
	'consider_incoming_relations': True,
	######################
	'sort_archetypes_by_relevance': False, 
	'minimise': False, 
}

OQA_OPTIONS = {
	'answer_horizon': 20,
	######################
	## QuestionAnswerer stuff
	'answer_pertinence_threshold': 0.15, 
	'tfidf_importance': 1/2,
	'answer_to_question_max_similarity_threshold': None,
	'answer_to_answer_max_similarity_threshold': None,
	'use_weak_pointers': False,
	# 'top_k': 100,
	'minimise': True, 

	'keep_the_n_most_similar_concepts': 2, 
	'query_concept_similarity_threshold': 0.75, 
	'add_external_definitions': False, 
	'include_super_concepts_graph': False, 
	'include_sub_concepts_graph': True, 
	'consider_incoming_relations': True,
}

QA_EXTRACTOR_OPTIONS = {
	'models_dir': 'question_extractor/data/models', 

	# 'sbert_model': {
	# 	'url': 'facebook-dpr-question_encoder-multiset-base', # model for paraphrase identification
	# 	'use_cuda': True,
	# },
	'tf_model': {
		# 'url': 'https://tfhub.dev/google/universal-sentence-encoder-qa2/3', # English QA
		'url': 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3', # Multilingual QA # 16 languages (Arabic, Chinese-simplified, Chinese-traditional, English, French, German, Italian, Japanese, Korean, Dutch, Polish, Portuguese, Spanish, Thai, Turkish, Russian)
		# 'url': 'https://tfhub.dev/google/LAReQA/mBERT_En_En/1',
		# 'cache_dir': '/Users/toor/Documents/Software/DLModels/tf_cache_dir/',
		# 'use_cuda': True,
	}, 

	# 'with_cache': False,
	'with_tqdm': True,
	'use_cuda': True,
	'default_batch_size': 10,
	'default_cache_dir': cache_path,
	'generate_kwargs': {
		"max_length": 256,
		"num_beams": 10,
		"num_return_sequences": 1,
		# "length_penalty": 1.5,
		# "no_repeat_ngram_size": 3, # do not set it when answer2question=False, questions always start with the same ngrams 
		"early_stopping": True,
	},
	'e2e_generate_kwargs': {
		"max_length": 256,
		"num_beams": 10,
		"num_return_sequences": 10,
		"num_beam_groups": 10,
		# "length_penalty": 1.5,
		# "no_repeat_ngram_size": 3, # do not set it when answer2question=False, questions always start with the same ngrams 
		# "return_dict_in_generate": True,
		# "forced_eos_token_id": True
		"early_stopping": True,
	},
	'task_list': [
		'answer2question', 
		'question2answer'
	],
}

QA_CLEANING_OPTIONS = {
	# 'sorted_template_list': None, 
	'min_qa_pertinence': 0.05, 
	'max_qa_similarity': 1, 
	'min_answer_to_sentence_overlap': 0.75,
	'min_question_to_sentence_overlap': 0, 
	'max_answer_to_question_overlap': 0.75,
	'coreference_resolution': False,
}

KG_MANAGER_OPTIONS = {
	# 'spacy_model': 'en_core_web_trf',
	# 'n_threads': 1,
	# 'use_cuda': True,
	'with_cache': True,
	'with_tqdm': True,

	'min_triplet_len': 0,
	'max_triplet_len': float('inf'),
	'min_sentence_len': 0,
	'max_sentence_len': float('inf'),
	'min_paragraph_len': 0,
	'max_paragraph_len': float('inf'),
}

GRAPH_EXTRACTION_OPTIONS = {
	'add_verbs': False, 
	'add_predicates_label': False, 
	'add_subclasses': True, 
	'use_wordnet': False,
}

GRAPH_BUILDER_OPTIONS = {
	'spacy_model': 'en_core_web_md',
	# 'n_threads': 1,
	# 'use_cuda': True,

	'with_cache': True,
	'with_tqdm': True,

	'max_syntagma_length': None,
	'add_source': True,
	'add_label': True,
	'lemmatize_label': False,

	'default_similarity_threshold': 0.75,
	'tf_model': {
		'url': 'https://tfhub.dev/google/universal-sentence-encoder-large/5', # Transformer
		# 'url': 'https://tfhub.dev/google/universal-sentence-encoder/4', # DAN
		# 'cache_dir': '/Users/toor/Documents/Software/DLModels/tf_cache_dir/',
	},
	'with_centered_similarity': True,
}

CONCEPT_CLASSIFIER_DEFAULT_OPTIONS = {
	'spacy_model': 'en_core_web_md',
	# 'n_threads': 1,
	# 'use_cuda': True,

	'tf_model': {
		'url': 'https://tfhub.dev/google/universal-sentence-encoder-large/5', # Transformer
		# 'url': 'https://tfhub.dev/google/universal-sentence-encoder/4', # DAN
		# 'cache_dir': '/Users/toor/Documents/Software/DLModels/tf_cache_dir/',
		# 'use_cuda': True,
		'with_cache': True,
		'with_tqdm': True,
	},
	'with_centered_similarity': True,
	'default_similarity_threshold': 0.75,
	# 'default_tfidf_importance': 3/4,

	'with_cache': True,
	'with_tqdm': True,
}

SENTENCE_CLASSIFIER_DEFAULT_OPTIONS = {
	'spacy_model': 'en_core_web_md',
	# 'n_threads': 1,
	# 'use_cuda': True,
	'with_centered_similarity': False,
	# 'with_topic_scaling': False,
	'with_stemmed_tfidf': True,
	'default_tfidf_importance': 1/2,

	'with_cache': True,
	'with_tqdm': True,
}

SUMMARISER_DEFAULT_OPTIONS = {
	# 'spacy_model': 'en_core_web_trf',
	# 'n_threads': 1,
	# 'use_cuda': True,

	'hf_model': {
		# 'url': 't5-base',
		'url': 'facebook/bart-large-cnn', # baseline
		# 'url': 'google/pegasus-billsum',
		# 'url': 'sshleifer/distilbart-cnn-12-6', # speedup (over the baseline): 1.24
		# 'url': 'sshleifer/distilbart-cnn-12-3', # speedup (over the baseline): 1.78
		# 'url': 'sshleifer/distilbart-cnn-6-6', # speedup (over the baseline): 2.09
		# 'cache_dir': '/Users/toor/Documents/Software/DLModels/hf_cache_dir/',
		'framework': 'pt',
		# 'use_cuda': True,
	},
}

################ Initialise data structures ################
graph_cache = os_path.join(cache_path,f"cache_graph_lemma-{GRAPH_BUILDER_OPTIONS['lemmatize_label']}.pkl")
edu_graph_cache = os_path.join(cache_path,f"cache_edu_graph.pkl")
edu_disco_only_graph_cache = os_path.join(cache_path,f"cache_edu_disco_only_graph.pkl")
edu_amr_only_graph_cache = os_path.join(cache_path,f"cache_edu_amr_only_graph.pkl")
# betweenness_centrality_cache = os_path.join(cache_path,'cache_betweenness_centrality.pkl')
qa_disco_cache = os_path.join(cache_path,'cache_qa_disco_embedder.pkl')

qa_edu_cache = os_path.join(cache_path,'cache_qa_edu_embedder.pkl')
qa_cache = os_path.join(cache_path,'cache_qa_embedder.pkl')

def init(info_type, model_type, tfidf_importance=None):
	print(f'server_interface {info_type} {model_type} tfidf_importance={tfidf_importance}')

	if tfidf_importance is not None:
		OQA_OPTIONS['tfidf_importance'] = tfidf_importance

	################ Configuration ################
	if info_type == 'tf':
		SENTENCE_CLASSIFIER_DEFAULT_OPTIONS['tf_model'] = {
			# 'url': 'https://tfhub.dev/google/universal-sentence-encoder-qa/3', # English QA
			'url': 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3', # Multilingual QA # 16 languages (Arabic, Chinese-simplified, Chinese-traditional, English, French, German, Italian, Japanese, Korean, Dutch, Polish, Portuguese, Spanish, Thai, Turkish, Russian)
			# 'url': 'https://tfhub.dev/google/LAReQA/mBERT_En_En/1',
			# 'cache_dir': '/Users/toor/Documents/Software/DLModels/tf_cache_dir/',
			'use_cuda': True,
		}
	elif info_type == 'minilm':
		SENTENCE_CLASSIFIER_DEFAULT_OPTIONS['sbert_model'] = {
			'url': 'multi-qa-MiniLM-L6-cos-v1', # model for paraphrase identification
			'use_cuda': True,
		}
	elif info_type == 'mpnet':
		SENTENCE_CLASSIFIER_DEFAULT_OPTIONS['sbert_model'] = {
			'url': 'multi-qa-mpnet-base-cos-v1', # model for paraphrase identification
			'use_cuda': True,
		}

	########################################################################
	print('Building Ontology Edge List..')
	graph = load_or_create_cache(
		graph_cache, 
		lambda: KnowledgeGraphBuilder(GRAPH_BUILDER_OPTIONS).set_documents_path(document_path, **GRAPH_CLEANING_OPTIONS).build(**GRAPH_EXTRACTION_OPTIONS)
	)
	########################################################################
	print('Building Question Answerer..')
	# betweenness_centrality = load_or_create_cache(
	# 	betweenness_centrality_cache, 
	# 	lambda: get_betweenness_centrality(filter(lambda x: '{obj}' in x[1], graph))
	# )

	qa_dict_list = load_or_create_cache(qa_disco_cache+'.qa_dict_list.pkl', lambda: QuestionAnswerExtractor(QA_EXTRACTOR_OPTIONS).extract(graph, use_paragraph_text=False))
	qa_dict_list = load_or_create_cache(qa_disco_cache+'.cleaned_qa_dict_list.pkl', lambda: QuestionAnswerExtractor(QA_EXTRACTOR_OPTIONS).clean_qa_dict_list(qa_dict_list, **QA_CLEANING_OPTIONS))
	# qa_dict_list = load_or_create_cache(qa_disco_cache+'.filtered_qa_dict_list.pkl', lambda: filter_invalid_sentences(QuestionAnswerExtractor(QA_EXTRACTOR_OPTIONS), qa_dict_list, key=lambda x: x['sentence'], avoid_coreferencing=False))
	edu_graph = []
	if 'edu_amr' in model_type:
		edu_graph = load_or_create_cache(
			edu_graph_cache, 
			lambda: QuestionAnswerExtractor(QA_EXTRACTOR_OPTIONS).extract_aligned_graph_from_qa_dict_list(
				KnowledgeGraphManager(KG_MANAGER_OPTIONS, graph), 
				qa_dict_list,
				GRAPH_BUILDER_OPTIONS, 
				use_paragraph_text=False,
				**GRAPH_CLEANING_OPTIONS,
			)
		)
	elif 'edu' in model_type:
		edu_graph = load_or_create_cache(
			edu_disco_only_graph_cache, 
			lambda: QuestionAnswerExtractor(QA_EXTRACTOR_OPTIONS).extract_aligned_graph_from_qa_dict_list(
				KnowledgeGraphManager(KG_MANAGER_OPTIONS, graph), 
				qa_dict_list,
				GRAPH_BUILDER_OPTIONS, 
				use_paragraph_text=False,
				qa_type_to_use= [
					'disco', # elementary discourse units
					# 'qaamr', # abstract meaning representations
				],
				**GRAPH_CLEANING_OPTIONS,
			)
		)
	elif 'amr' in model_type:
		edu_graph = load_or_create_cache(
			edu_amr_only_graph_cache, 
			lambda: QuestionAnswerExtractor(QA_EXTRACTOR_OPTIONS).extract_aligned_graph_from_qa_dict_list(
				KnowledgeGraphManager(KG_MANAGER_OPTIONS, graph), 
				qa_dict_list,
				GRAPH_BUILDER_OPTIONS, 
				use_paragraph_text=False,
				qa_type_to_use= [
					# 'disco', # elementary discourse units
					'qaamr', # abstract meaning representations
				],
				**GRAPH_CLEANING_OPTIONS,
			)
		)
	del qa_dict_list
	if 'clause' in model_type:
		kg = list(unique_everseen(edu_graph + graph))
		save_graphml(kg, 'knowledge_graph')
		kg_manager = KnowledgeGraphManager(KG_MANAGER_OPTIONS, kg)
		print('Graph size:', len(kg))
		print('Grammatical clauses:', len(list(filter(lambda x: '{obj}' in x[1], kg))))
		print('Avg clause len:', np.mean(list(map(len, map(kg_manager.get_label, unique_everseen(map(lambda x: x[1], filter(lambda x: '{obj}' in x[1], kg))))))))
		del kg
	else:
		save_graphml(edu_graph, 'knowledge_graph')
		kg_manager = KnowledgeGraphManager(KG_MANAGER_OPTIONS, edu_graph)
		print('Graph size:', len(edu_graph))
		print('Grammatical clauses:', len(list(filter(lambda x: '{obj}' in x[1], edu_graph))))
		print('Avg clause len:', np.mean(list(map(len, map(kg_manager.get_label, unique_everseen(map(lambda x: x[1], filter(lambda x: '{obj}' in x[1], edu_graph))))))))
	del edu_graph
	del graph
	
	qa = QuestionAnswerer(
		kg_manager= kg_manager,
		concept_classifier_options= CONCEPT_CLASSIFIER_DEFAULT_OPTIONS,
		sentence_classifier_options= SENTENCE_CLASSIFIER_DEFAULT_OPTIONS,
		answer_summariser_options= SUMMARISER_DEFAULT_OPTIONS,
		# betweenness_centrality=None, 
	)
	qa.load_cache(qa_edu_cache, save_if_init=True)
	return qa

################ Define methods ################
def get_question_answer_dict(qa, question_list, options=None):
	if not options:
		options = {}
	question_answer_dict = qa.ask(question_list, **options)
	# for k,v in question_answer_dict.items():
	# 	question_answer_dict[k] = remove_similar_labels(v, key=lambda x: x['sentence'])
	# print('######## Question Answers ########')
	# print(json.dumps(question_answer_dict, indent=4))
	# qa.store_cache(qa_edu_cache)
	return question_answer_dict

def get_question_answer_dict_quality(qa, question_answer_dict, top=5):
	return qa.get_question_answer_dict_quality(question_answer_dict, top=top)

def get_summarised_question_answer_dict(qa, question_answer_dict, options=None):
	if not options:
		options = {}
	question_summary_tree = qa.summarise_question_answer_dict(question_answer_dict, **options)
	return question_summary_tree

def get_concept_overview(qa, query_template_list=None, concept_uri=None, concept_label= None, options=None):
	if not options:
		options = {}
	# set consider_incoming_relations to False with concept-centred generic questions (e.g. what is it?), otherwise the answers won't be the sought ones
	question_answer_dict = qa.get_concept_overview(
		query_template_list = query_template_list, 
		concept_uri = concept_uri,
		concept_label = concept_label,
		**options
	)
	# print('######## Concept Overview ########')
	# print(concept_uri, json.dumps(question_summarised_answer_dict, indent=4))
	store_cache()
	return question_answer_dict

def annotate_text(qa, sentence, similarity_threshold=None, max_concepts_per_alignment=1):
	return qa.concept_classifier.annotate(
		DocParser().set_content_list([sentence]), 
		similarity_threshold=similarity_threshold, 
		max_concepts_per_alignment=max_concepts_per_alignment,
		concept_id_filter=lambda x: x in qa.important_aspect_set,
	)

def annotate_question_summary_tree(qa, question_summary_tree, similarity_threshold=None, max_concepts_per_alignment=1):
	return qa.annotate_question_summary_tree(question_summary_tree, similarity_threshold=similarity_threshold, max_concepts_per_alignment=max_concepts_per_alignment)

def get_taxonomical_view(qa, concept_uri, depth=0):
	return qa.get_taxonomical_view(concept_uri, depth=depth)

def annotate_taxonomical_view(qa, taxonomical_view, similarity_threshold=None, max_concepts_per_alignment=1):
	return qa.annotate_taxonomical_view(taxonomical_view, similarity_threshold=similarity_threshold, max_concepts_per_alignment=max_concepts_per_alignment)

def get_equivalent_concepts(qa, concept_uri):
	return qa.adjacency_list.get_equivalent_concepts(concept_uri)

def store_cache(qa):
	qa.store_cache(qa_edu_cache)
	# qa.store_cache(qa_cache)

# ############### Cache scheduler ###############
# SCHEDULING_TIMER = 15*60 # 15 minutes
# from threading import Timer
# def my_task(is_first=False):
# 	if not is_first:
# 		store_cache()
# 	Timer(SCHEDULING_TIMER, my_task).start()
# # start your scheduler
# my_task(is_first=True)
