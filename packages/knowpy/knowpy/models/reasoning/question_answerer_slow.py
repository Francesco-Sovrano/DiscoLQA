from knowpy.models.reasoning.question_answerer import QuestionAnswerer
from knowpy.models.knowledge_extraction.question_answer_extractor import QuestionAnswerExtractor
from knowpy.models.knowledge_extraction.ontology_builder import OntologyBuilder
from knowpy.misc.levenshtein_lib import remove_similar_labels
from more_itertools import unique_everseen
from knowpy.misc.jsonld_lib import *

class QuestionAnswererSlow(QuestionAnswerer):

	def get_sourced_graph_from_aspect_graph(self, concept_graph, add_clustered_triples=False, **args):
		# Get labeled triples
		label_graph = list(self.kg_manager.get_labeled_graph_from_concept_graph(concept_graph))
		# Add clustered triples
		if add_clustered_triples:
			label_graph += self.find_couple_clusters_in_labeled_graph(label_graph)
		# Add source to triples
		return self.kg_manager.get_sourced_graph_from_labeled_graph(label_graph)

	def find_couple_clusters_in_labeled_graph(self, label_graph):
		sp_dict = {}
		po_dict = {}
		couple_clusters = []
		for labeled_triple, original_triple in label_graph:
			sp_key = (tuple(labeled_triple[:2]),tuple(original_triple[:2]))
			o_list = sp_dict.get(sp_key, None)
			if not o_list:
				o_list = sp_dict[sp_key] = []
			o_list.append((labeled_triple[-1],original_triple[-1]))
			po_key = (tuple(labeled_triple[1:]),tuple(original_triple[1:]))
			s_list = po_dict.get(po_key, None)
			if not s_list:
				s_list = po_dict[po_key] = []
			s_list.append((labeled_triple[0],original_triple[0]))

		for sp_key,o_list in sp_dict.items():
			if len(o_list) <= 1:
				continue
			labeled_sp, original_sp = sp_key
			o_list = unique_everseen(sorted(o_list, key=lambda x: x[0]), key=lambda x: x[-1])
			o_list = remove_similar_labels(list(o_list))
			labeled_o, original_o = zip(*o_list)
			couple_clusters.append((
				(*labeled_sp, tuple(labeled_o)),
				(*original_sp, tuple(original_o))
			))
		for po_key,s_list in po_dict.items():
			if len(s_list) <= 1:
				continue
			labeled_po, original_po = po_key
			s_list = unique_everseen(sorted(s_list, key=lambda x: x[0]), key=lambda x: x[-1])
			s_list = remove_similar_labels(list(s_list))
			labeled_s, original_s = zip(*s_list)
			couple_clusters.append((
				(tuple(labeled_s), *labeled_po),
				(tuple(original_s), *original_po)
			))
		return couple_clusters
