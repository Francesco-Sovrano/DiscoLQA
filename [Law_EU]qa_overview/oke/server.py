import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from bottle import run, get, post, route, hook, request, response, static_file
import json
import numpy as np

from more_itertools import unique_everseen
import sys
import logging
logger = logging.getLogger('knowpy')
# logger.setLevel(logging.INFO)
logger.setLevel(logging.ERROR)
logger.addHandler(logging.StreamHandler(sys.stdout))

global model_type
global info_type
_,port,model_type,info_type = sys.argv
port = int(port)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

model_type = model_type.casefold()
info_type = info_type.casefold()
print(f'server {info_type} {model_type}')
from server_interface import *
qa = init(model_type,info_type)

###############################################################
# CORS

@route('/<:re:.*>', method='OPTIONS')
def enable_cors_generic_route():
	"""
	This route takes priority over all others. So any request with an OPTIONS
	method will be handled by this function.

	See: https://github.com/bottlepy/bottle/issues/402

	NOTE: This means we won't 404 any invalid path that is an OPTIONS request.
	"""
	add_cors_headers()

@hook('after_request')
def enable_cors_after_request_hook():
	"""
	This executes after every route. We use it to attach CORS headers when
	applicable.
	"""
	add_cors_headers()

def add_cors_headers():
	try:
		response.headers['Access-Control-Allow-Origin'] = '*'
		response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, OPTIONS'
		response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'
	except Exception as e:
		print('Error:',e)

def get_from_cache(cache, key, build_fn):
	if key not in cache:
		cache[key] = json.dumps(build_fn())
	return cache[key]

###############################################################
# API - Question Answerer

ANSWERS_CACHE = {}
@get('/answer')
def get_answer():
	response.content_type = 'application/json'
	# question = request.forms.get('question') # post
	user_id = request.query.get('uuid')
	question = request.query.get('question')
	if not question.endswith('?'):
		question += '?'
	def build_fn():
		print(f'Answering to {user_id}..')
		# print(question)
		question_answer_dict = get_question_answer_dict(
			qa,
			[question],
			options= OQA_OPTIONS
		)
		print('Summarising..')
		if not question_answer_dict:
			return None
		tree_arity = 1
		question_summary_tree = get_summarised_question_answer_dict(
			qa,
			question_answer_dict,
			options={
				'ignore_non_grounded_answers': False, 
				'use_abstracts': False, 
				'summary_horizon': OQA_OPTIONS['answer_horizon'],
				'tree_arity': tree_arity, 
				# 'cut_factor': 2, 
				'depth': 1,
				'remove_duplicates': False,
				'min_size_for_summarising': float('inf'),
			}
		)
		print('Annotating..')
		return {
			'question_summary_tree': question_summary_tree,
			'annotation_list': annotate_question_summary_tree(qa, question_summary_tree, is_preprocessed_content=True) if WITH_ANNOTATIONS else [],
			'quality': get_question_answer_dict_quality(qa, question_answer_dict, top=tree_arity),
		}
	return get_from_cache(ANSWERS_CACHE, question, build_fn)

OVERVIEW_CACHE = {}
@get('/overview')
def get_overview():
	response.content_type = 'application/json'
	user_id = request.query.get('uuid')
	concept_uri = request.query.get('concept_uri')
	# concept_uri = concept_uri.lower().strip()
	# query_template_list = json.loads(request.query.get('query_template_list'))
	def build_fn():
		print(f'Answering to {user_id}..')
		query_template_list = [
			'What is {X}?',
			'Who {X}?',
			'Why {X}?',
			'How {X}?',
			'Where {X}?',
			'When {X}?',
			'Which {X}?',
			'Whose {X}?',
		]
		question_answer_dict = get_concept_overview(
			qa, 
			concept_uri= concept_uri, 
			options= OVERVIEW_OPTIONS,
		)
		# # Normalize confidence scores: max is assumed to be 0.1
		# for formatted_answer_list in question_answer_dict.values():
		# 	for answer_dict in formatted_answer_list:
		# 		answer_dict['confidence'] = min(1.,answer_dict['confidence']/0.1)
		print('Summarising..')
		if question_answer_dict:
			question_summary_tree = get_summarised_question_answer_dict(
				qa, 
				question_answer_dict,
				options={
					'ignore_non_grounded_answers': False, 
					'use_abstracts': False, 
					'summary_horizon': OVERVIEW_OPTIONS['answer_horizon'],
					'tree_arity': 1, 
					# 'cut_factor': 1, 
					'depth': 1,
					'remove_duplicates': False,
					'min_size_for_summarising': float('inf'),
				}
			)
		else:
			question_summary_tree = None
		print('Getting taxonomical view..')
		taxonomical_view = get_taxonomical_view(qa, concept_uri, depth=0)
		print('Annotating..')
		annotation_iter = unique_everseen(annotate_question_summary_tree(qa, question_summary_tree) + annotate_taxonomical_view(qa, taxonomical_view))
		equivalent_concept_uri_set = get_equivalent_concepts(qa, concept_uri)
		equivalent_concept_uri_set.add(concept_uri)
		annotation_iter = filter(lambda x: x['annotation'] not in equivalent_concept_uri_set, annotation_iter)
		return {
			'question_summary_tree': question_summary_tree,
			'taxonomical_view': taxonomical_view,
			'annotation_list': list(annotation_iter),
		}
	return get_from_cache(OVERVIEW_CACHE, concept_uri, build_fn)

ANNOTATION_CACHE = {}
@get('/annotation')
def get_annotation():
	response.content_type = 'application/json'
	# question = request.forms.get('question') # post
	sentence = request.query.get('sentence')
	is_preprocessed_content = request.query.get('is_preprocessed_content', True)
	def build_fn():
		print('Annotating..')
		return annotate_text(qa, sentence, max_concepts_per_alignment=1, is_preprocessed_content=True)
	return get_from_cache(ANNOTATION_CACHE, sentence, build_fn)

if __name__ == "__main__":
	run(host='0.0.0.0', port=port+2, debug=True)
	