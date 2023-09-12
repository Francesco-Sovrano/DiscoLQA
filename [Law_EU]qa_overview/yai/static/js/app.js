const OKE_SERVER_URL = location.protocol+'//'+location.hostname+(location.port ? ':'+(parseInt(location.port,10)+2): '')+'/';
console.log('OKE_SERVER_URL:', OKE_SERVER_URL);
const GET_OVERVIEW_API = OKE_SERVER_URL+"overview";
const GET_ANSWER_API = OKE_SERVER_URL+"answer";
const GET_ANNOTATION_API = OKE_SERVER_URL+"annotation";

OVERVIEW_CACHE = {};
TAXONOMICAL_VIEW_CACHE = {};
ANNOTATION_CACHE = {};
ANNOTATED_HTML_CACHE = {};
KNOWN_KNOWLEDGE_GRAPH = [
	{
		'@id': 'myf:gdpr.akn',
		'@type': 'dbr:Regulation',
		'my:url': [
			'https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32016R0679'
		], 
		'rdfs:label': 'GDPR: Regulation EU No 2016/679'
	},
	{
		'@id': 'myf:eidas.akn',
		'@type': 'dbr:Regulation',
		'my:url': [
			'https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32014R0910'
		],
		'rdfs:label': 'eIDAS: Regulation EU No 910/2014'
	},
	{
		'@id': 'myf:warrant.html',
		'@type': 'dbr:CouncilFrameworkDecision',
		'my:url': [
			'https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:02002F0584-20090328&from=EN',
		], 
		'rdfs:label': 'COUNCIL FRAMEWORK DECISION of 13 June 2002 on the European arrest warrant and the surrender procedures between Member States'
	},
	{
		'@id': 'myf:bruss.html',
		'@type': 'dbr:Regulation',
		'my:url': [
			'https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32012R1215&from=EN'
		], 
		'rdfs:label': 'Brussels I bis Regulation EU 1215/2012'
	},
	{
		'@id': 'myf:rome_i.html',
		'@type': 'dbr:Regulation',
		'my:url': [
			'https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32008R0593&from=EN'
		],
		'rdfs:label': 'Rome I Regulation EC 593/2008'
	},
	{
		'@id': 'myf:rome_ii.html',
		'@type': 'dbr:Regulation',
		'my:url': [
			'https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32007R0864&from=EN'
		],
		'rdfs:label': 'Rome II Regulation EC 864/2007'
	}
];
KNOWN_KNOWLEDGE_GRAPH = format_jsonld(KNOWN_KNOWLEDGE_GRAPH);
KNOWN_ENTITY_DICT = get_typed_entity_dict_from_jsonld(KNOWN_KNOWLEDGE_GRAPH);

const app = new Vue({
	el: '#app',
	data: {
		overview_api: GET_OVERVIEW_API,
		answer_api: GET_ANSWER_API,
		
		question_input_placeholder: 'Write a question.. e.g. Which law is applicable to a non-contractual obligation?',
		question_input_default_value: 'Which law is applicable to a non-contractual obligation?',

		documents: [
			'myf:gdpr.akn', // data protection - GDPR 
			'myf:eidas.akn', // electronic identification - eIDAS
			'myf:warrant.html', // EU arrest warrant
			'myf:bruss.html', // obligations - Brussels
			'myf:rome_i.html', // obligations - Rome I
			'myf:rome_ii.html', // obligations - Rome II
		].map(x=>template_expand(get_known_label(x), x)),
	}
})
