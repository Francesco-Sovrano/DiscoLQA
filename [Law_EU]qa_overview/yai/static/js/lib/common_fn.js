const DEBUG = false;
const DBPEDIA_ENDPOINT = "//dbpedia.org/sparql";

const PREFIX_MAP = {
	'my': 'http://my_graph.co/',
	'myfile': 'http://my_graph.co/files/',
	'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
	'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
	'prov': 'http://www.w3.org/ns/prov#',
	'foaf': 'http://xmlns.com/foaf/0.1/',
	'skos': 'http://www.w3.org/2004/02/skos/core#',
	'dct': 'http://purl.org/dc/terms/',
	'dc': 'http://purl.org/dc/terms/',
	'dce': 'http://purl.org/dc/elements/1.1/',
	'owl': 'http://www.w3.org/2002/07/owl#',
	'dbo': 'http://dbpedia.org/ontology/',
	'dbp': 'http://dbpedia.org/property/',
	'dbr': 'http://dbpedia.org/resource/',
	'ns1': 'http://purl.org/linguistics/gold/',
	'vrank': 'http://purl.org/voc/vrank#',
	'wn': 'http://wordnet.princeton.edu/',
	'brusselsreg_en_1215-20212': 'http://my_graph.co/brusselsreg_en_1215-20212/',
	'rome_i_en': 'http://my_graph.co/rome_i_en/',
	'rome_ii_en': 'http://my_graph.co/rome_ii_en/',
}
const PREFIX_MAP_STRING = Object.keys(PREFIX_MAP).map(x => "PREFIX "+x+": <"+PREFIX_MAP[x]+">").join("\n");

function prefixed_string_to_uri(prefixed_string)
{
	prefixed_string = String(prefixed_string);
	if (prefixed_string == '@type')
		return PREFIX_MAP['rdf']+'type';
	var item_list = prefixed_string.split(':');
	if (item_list.length>1 && item_list[0] in PREFIX_MAP)
		return PREFIX_MAP[item_list[0]] + item_list.slice(1).join(':');
	return prefixed_string;
};

function uri_to_prefixed_string(uri)
{
	var prefixed_string = uri;
	for (var [id,url] of Object.entries(PREFIX_MAP).sort((a,b)=>b[0].length-a[0].length))
		prefixed_string = prefixed_string.replace(url,id+':');
	return prefixed_string;
};

const TYPE_URI = prefixed_string_to_uri('rdf:type');
const SUBCLASSOF_URI = prefixed_string_to_uri('rdfs:subClassOf');
const LABEL_URI = prefixed_string_to_uri('rdfs:label');

const HAS_ENTITY_URI = prefixed_string_to_uri('my:hasEntity');
const HAS_SUBCLASS_URI = prefixed_string_to_uri('my:hasSubClass');
const UNKNOWN_TYPE_URI = prefixed_string_to_uri('my:Unknown');
const TYPESET_URI = prefixed_string_to_uri('my:typeSet');
const ENTITY_PERCENTAGE_URI = prefixed_string_to_uri('my:presenceInDataset');
const ENTITY_COUNT_URI = prefixed_string_to_uri('my:entityCount');
const CLASS_COUNT_URI = prefixed_string_to_uri('my:classCount');
const STATEMENT_COUNT_URI = prefixed_string_to_uri('my:statementCount');
const CLASS_LIST_URI = prefixed_string_to_uri('my:hasClass');
const DOCUMENT_TITLE_URI = prefixed_string_to_uri('my:documentTitle');
const DATASET_LIST_URI = prefixed_string_to_uri('my:datasetList');
const PROCESS_LIST_URI = prefixed_string_to_uri('my:processList');
const IS_COMPOSITE_CLASS_BOOL_URI = prefixed_string_to_uri('my:isCompositeClass');
const COMPOSITE_CLASS_SET_URI = prefixed_string_to_uri('my:classSet');
const EP_OVERVIEW_URI = prefixed_string_to_uri('my:explanatoryProcessOverview');
const ANNOTATION_LIST_URI = prefixed_string_to_uri('my:annotationList');
const WORD_ANNOTATION_LIST_URI = prefixed_string_to_uri('my:wordLevelAnnotationList');
const ANNOTATED_SENTENCE_LIST_URI = prefixed_string_to_uri('my:annotatedSentenceList');
const RELATED_TO_URI = prefixed_string_to_uri('my:relatedTo');
const TEXT_URI = prefixed_string_to_uri('my:text');
const PROCESS_INPUT_URI = prefixed_string_to_uri('my:process_input');
const FEATURE_URI = prefixed_string_to_uri('my:feature');
const VALUE_URI = prefixed_string_to_uri('my:value');
const FEATURE_ORDER_URI = prefixed_string_to_uri('my:feature_order');
const COUNTERFACTUAL_API_URI = prefixed_string_to_uri('my:counterfactual_api_url');
const RELEVANT_PROCESS_INPUT_LIST_URI = prefixed_string_to_uri('my:relevant_process_input_list');
const PROPERTY_LIST_URI = prefixed_string_to_uri('my:propertyList');

function build_RDF_item(item, ground=null, source=null) {
	return {
		'@value': prefixed_string_to_uri(item),
		'@ground': ground,
		'@source': source,
	}
}

function isRDFItem(v)
{
	return typeof v==='object' && v!==null && v.constructor.name == "Object" && ('@value' in v);
};

function isArray(v) 
{
	return typeof v==='object' && v!==null && v.constructor.name == "Array";
};

function isDict(v) 
{
	return typeof v==='object' && v!==null && v.constructor.name == "Object" && !isRDFItem(v);
};

function isHTML(str)
{
	var html_pattern = "<(?:\"[^\"]*\"['\"]*|'[^']*'['\"]*|[^'\">])+>";
	var pattern = new RegExp(html_pattern,'i');
	return !!pattern.test(str);
};

function isNumber(str)
{
	return !isNaN(parseInt(str, 10));
}

function isURL(str) 
{
	if (!str)
		return false;
	if (isHTML(str))
		return false;
	if (str.startsWith('../') || str.startsWith('./'))
		return true;
	var url_pattern = '<?(http[s]?:)?//(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+>?';
	var pattern = new RegExp(url_pattern,'i');
	return !!pattern.test(str);
};

function decodeHtmlEntity(encodedString) 
{
	var textArea = document.createElement('textarea');
	textArea.innerHTML = encodedString;
	var object = textArea.value;
	object = String(object).trim()
	object = object.replace(/<|>|"/gi,'')
	return object
}

function pathJoin(parts, sep) 
{
	const separator = sep || '/';
	parts = parts.map((part, index)=>{
		part = String(part);
		 if (index)
			part = part.replace(new RegExp('^' + separator), '');
		 if (index !== parts.length - 1)
			part = part.replace(new RegExp(separator + '$'), '');
		 return part;
	})
	parts = parts.filter(x => x!='');
	return parts.join(separator);
};

function getPath(url)
{
	var reUrlPath = /(?:\w+:)?\/\/[^\/]+(\/.+)*(\/.+)/;
	var urlParts = url.match(reUrlPath) || [url, url];
	path = (urlParts.length > 1) ? urlParts[urlParts.length-1].replace(/\//gi,'') : url;
	if (path.includes('#'))
	{
		splitted_path = path.split('#');
		path = splitted_path[splitted_path.length-1];
	}
	return path;
}

function format_string(v, toLowerCase=true) 
{
	var formatted_string = String(v).replace(/([a-z])([A-Z0-9]+)/g, '$1 $2').replace(/[_ ]+/g, ' ');
	return toLowerCase?formatted_string.toLowerCase():formatted_string;
}

function format_link(link, toLowerCase=true)
{
	return format_string(getPath(link), toLowerCase);
}

function template_expand(name,topic=null,label=null) 
{
	if (!topic)
		topic = name;
	if (!label)
		label = name;
	return `<annotation><span 
				class="link"
				data-topic="${topic}"
				data-label="${label}"
			>${name}</span></annotation>`;
}

function linkify(link, name=null, label=null)
{
	link = String(link).replace(/<|>|"/gi,'');
	if (!name)
		name = isURL(link) ? format_link(link) : link;

	return template_expand(name,link,label);
}

function HTMLescape(s) {
    return s.replace(/&/g, '&amp;')
            .replace(/"/g, '&quot;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
}

function remove_anonymous_entities(data, entity_dict) 
{
	if (isArray(data))
		return data.map(x=>remove_anonymous_entities(x, entity_dict));

	if (isDict(data)) 
	{
		for (var [p,o] of Object.entries(data)) 
		{
			if (p=='@id')
				continue;
			data[p] = remove_anonymous_entities(o, entity_dict);
		}
	}
	else // is RDF item
	{
		var data_desc = get_RDFItem_description(data);
		if (data_desc.startsWith('_:') && data_desc in entity_dict) // anonymous entity
			return entity_dict[data_desc];
	}
	return data;
}

function tuple_list_to_formatted_jsonld(tuple_list)
{
	var jsonld = [];
	for (var [s,p,o] of tuple_list)
	{
		var triple_jsonld = {'@id': s};
		triple_jsonld[p] = o;
		jsonld.push(triple_jsonld);
	}
	// Remove anonymous entities, if possible
	jsonld = format_jsonld(jsonld);
	// return remove_anonymous_entities(
	// 	jsonld,
	// 	get_entity_dict(build_minimal_entity_graph(jsonld))
	// );
	return jsonld;
}

function format_jsonld(jsonld, ground=null, source=null)
{
	if (isArray(jsonld))
		return jsonld.map(x=>format_jsonld(x, ground, source));
	if (!isDict(jsonld))
		return isRDFItem(jsonld)?build_RDF_item(jsonld['@value'], ground, jsonld['@source']):build_RDF_item(jsonld, ground, source)
	// format id
	if ('url' in jsonld && !('@id' in jsonld)) {
		jsonld['@id'] = jsonld['url']
		jsonld['url'] = null
		delete jsonld['url']
	}
	// format predicates
	var new_jsonld = {}
	for (var [predicate, object_list] of Object.entries(jsonld))
	{
		if (isArray(object_list) && object_list.length==0)
			continue
		var uri_predicate = prefixed_string_to_uri(predicate);
		new_jsonld[uri_predicate] = format_jsonld(object_list, ground, source);
	}
	return new_jsonld
}

function replace_jsonld_by_id(jsonld, jsonld_fragment)
{
	if (!('@id' in jsonld_fragment))
		return jsonld;

	if (isArray(jsonld))
		jsonld = jsonld.map(x=>replace_jsonld_by_id(x, jsonld_fragment));
	else if (isDict(jsonld))
	{
		for (var [predicate, object] of Object.entries(jsonld))
			jsonld[predicate] = replace_jsonld_by_id(object, jsonld_fragment);
	}
	else
	{
		if (jsonld['@value'] == jsonld_fragment['@id']['@value'])
			return jsonld_fragment;
	}
	return jsonld;
}

function get_value_in_jsonld_by_key(jsonld, key)
{
	if (isRDFItem(jsonld))
		return [];

	var value_list = [];
	if (isArray(jsonld))
	{
		for (var e of jsonld)
			value_list = value_list.concat(get_value_in_jsonld_by_key(e, key));
	}
	else
	{
		if (key in jsonld)
			return [jsonld[key]];
		if (prefixed_string_to_uri(key) in jsonld)
			return [jsonld[prefixed_string_to_uri(key)]];

		for (var [predicate, object] of Object.entries(jsonld))
			value_list = value_list.concat(get_value_in_jsonld_by_key(object, key));
	}
	return value_list;
}

function zip(arrays) {
    return arrays[0].map(function(_,i){
        return arrays.map(array=>array[i])
    });
}

function download(content, fileName, contentType) {
    var a = document.createElement("a");
    var file = new Blob([content], {type: contentType});
    a.href = URL.createObjectURL(file);
    a.download = fileName;
    a.click();
}

function query_sparql_endpoint(endpoint, queryStr, isDebug=false) 
{
	try {
		var querypart = "query=" + escape(queryStr);
		// Get our HTTP request object.
		var xmlhttp = null;
		if(window.XMLHttpRequest) 
			xmlhttp = new XMLHttpRequest();
	  	else if(window.ActiveXObject) // Code for older versions of IE, like IE6 and before.
			xmlhttp = new ActiveXObject("Microsoft.XMLHTTP");
		else 
			alert('Perhaps your browser does not support XMLHttpRequests?');

		// Set up a POST with JSON result format. GET can have caching probs, so POST
		xmlhttp.open('POST', endpoint, false); // `false` makes the request synchronous
		xmlhttp.setRequestHeader('Content-type', 'application/x-www-form-urlencoded');
		xmlhttp.setRequestHeader("Accept", "application/sparql-results+json");

		// Send the query to the endpoint.
		xmlhttp.send(querypart);
		if(xmlhttp.readyState == 4) 
		{
			if(isDebug)
				alert("Sparql query error: " + xmlhttp.status + " " + xmlhttp.responseText);
			if(xmlhttp.status == 200) 
				return eval('(' + xmlhttp.responseText + ')');
		}
	}
	catch(e)
	{
		if (DEBUG)
			console.error(e)
	}
	return null
};

function get_array_description(dict_list, limit=null)
{
	return dict_list.slice(0, limit).map(y=>linkify(get_description(y, as_label=false),titlefy(get_known_label(get_description(y, as_label=false))))).join(', ');
}

function get_dict_description(dict, as_label=true)
{
	if (!dict)
		return '';
	var label = '';
	// console.log(dict);
	if (as_label && LABEL_URI in dict && isRDFItem(dict[LABEL_URI]))
		label = dict[LABEL_URI]['@value'];
	else if ('@id' in dict)
		label = dict['@id']['@value'];
	return HTMLescape(String(label).trim());
}

function get_RDFItem_description(item)
{
	if (!item || !('@value' in item))
		return null;
	return HTMLescape(String(item['@value']).trim());
}

function get_description(e, as_label=true, limit=null)
{
	if (isRDFItem(e))
		return get_RDFItem_description(e);
	if (isDict(e))
		return get_dict_description(e, as_label);
	if (isArray(e))
		return get_array_description(e, limit);
	return e;
}

function isInt(value) {
  if (isNaN(value)) {
    return false;
  }
  var x = parseFloat(value);
  return (x | 0) === x;
}

function get_unique_elements(list, id_fn=x=>x) {
  var j = {};

  list.forEach( function(v) {
    j[id_fn(v)] = v;
  });

  return Object.values(j);
}

jQuery.extend(jQuery.expr[':'], {
  shown: function (el, index, selector) {
    return $(el).css('visibility') != 'hidden' && $(el).css('display') != 'none' && !$(el).is(':hidden')
  }
});

INLINE_TAG_LIST = [
	'a',
	'abbr',
	'acronym',
	'annotation',
	'b',
	'bdo',
	'big',
	'br',
	'button',
	'cite',
	'code',
	'dfn',
	'em',
	'i',
	'img',
	'input',
	'kbd',
	'label',
	'map',
	'object',
	'output',
	'q',
	'samp',
	'script',
	'select',
	'small',
	'span',
	'strong',
	'sub',
	'sup',
	'textarea',
	'time',
	'tt',
	'var'
].map(x=>x.toUpperCase());

function htmlDecode(input){
  var e = document.createElement('textarea');
  e.innerHTML = input;
  // handle case of empty input
  return e.childNodes.length === 0 ? "" : e.childNodes[0].nodeValue;
}

function removeAllAttrs(element) {
    for (var i= element.attributes.length; i-->0;)
        element.removeAttributeNode(element.attributes[i]);
}

function escapeRegExp(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); // $& means the whole matched string
}

const proximity_chars_regexp = /[^a-zA-Z0-9]/gi;
const remove_orphans = y=>y.replace(/ ['"()\[\]{}] /g,' ')
const tokenise = y=>y.split(proximity_chars_regexp).map(x=>x.toLowerCase())//.filter(x=>x!='');

function annotate_html(html, inner_text, annotation_list, annotate_fn, merge_span=false)
{
	if (!html)
		return html;
	if (!annotation_list)
		return remove_orphans(html);
	inner_text = tokenise(inner_text.replace(/[\n ]+/g,' ').replace(/[^ ]- +/g,'').trim()).join(' ').replace(/ +/g,' ');
	// console.log(inner_text);
	// console.log(JSON.stringify(annotation_list, null, 4));
	var annotated_html = remove_orphans(htmlDecode(html).replace(/ +/g,' ').trim()); // get decoded html
	for (const annotation_dict of annotation_list) // clean
		annotation_dict.text = remove_orphans(annotation_dict.text.replace(/ ([^a-zA-Z0-9]) /gi, '$1 ').trim());
	annotation_list = get_unique_elements(annotation_list, x=>x.text); // remove duplicates
	annotation_list = annotation_list.sort((a,b)=>b.text.length-a.text.length); // descending order
	// var annotation_text_list = annotation_list.map(x=>x.text);
	for (const annotation_dict of annotation_list)
	{
		var token_list = tokenise(annotation_dict.text);
		
		const cleaned_sub_text = token_list.join(' ');
		const regexp = new RegExp(escapeRegExp(cleaned_sub_text).trim(), 'gi');
		if (inner_text.search(regexp) < 0)
			continue;

		const annotation_uri = annotation_dict['annotation'];
		const last_token = token_list[token_list.length-1];
		var start_idx = annotated_html.length;
		const splitted_old_html = annotated_html.split('>');
		while (start_idx >= 0)
		{
			const splitted_html = annotated_html.slice(0,start_idx).split('>');
			var added = false;
			for (var i=splitted_html.length-1; !added && i>=0; --i)
			{
				const splitted_html_i = splitted_html[i];
				start_idx -= splitted_html_i.length+1;

				if (splitted_html_i.endsWith('/a')) // do not annotate anchors
					continue;

				const content = splitted_html_i.split('<')[0];
				const splitted_content = tokenise(content);
				const joined_splitted_content = splitted_content.join(' ');
				const label_is_in_content = joined_splitted_content.search(regexp) >= 0;
				
				var left_proxy_content = '';
				var right_proxy_content = '';
				const inline_tags_regexp = /^<\/?(span|annotation)/;
				if (!label_is_in_content && merge_span) {
					if (content == '')
						continue;
					// Proceed backward
					var j = i-1;
					var stop_iterating = false;
					while (!stop_iterating && j >= 0) {
						const jth_html_piece = splitted_old_html[j];
						--j;

						// if (jth_html_piece.startsWith('<span'))
						// 	continue;
						if (inline_tags_regexp.test(jth_html_piece))
							continue;
						// if (jth_html_piece.startsWith('<br'))
						// 	continue;
						var new_content = jth_html_piece.startsWith('<br')?' ':jth_html_piece.split('<')[0];
						// if (content.includes('sex'))
						// 	console.log('--j', new_content, jth_html_piece, splitted_old_html[j]);
						// proxy_content = new_content+proxy_content;
						left_proxy_content = new_content+left_proxy_content;
						if (new_content=='' || (new_content!=' ' && new_content.includes(' ')))
							stop_iterating = true;
					}
					// Proceed forward
					var j = i+1;
					var stop_iterating = false;
					while (!stop_iterating && j < splitted_old_html.length) {
						const jth_html_piece = splitted_old_html[j];
						++j;
						// console.log('++j', jth_html_piece);
						
						// if (jth_html_piece.startsWith('<span'))
						// 	continue;
						if (inline_tags_regexp.test(jth_html_piece))
							continue;
						// if (jth_html_piece.startsWith('<br'))
						// 	continue;
						var new_content = jth_html_piece.startsWith('<br')?' ':jth_html_piece.split('<')[0];
						// if (content.includes('sex'))
						// 	console.log('++j', new_content, jth_html_piece, splitted_old_html[j]);
						right_proxy_content = right_proxy_content+new_content;
						if (new_content=='' || (new_content != ' ' && new_content.includes(' ')))
							stop_iterating = true;
					}
					left_proxy_content = left_proxy_content.replace('- ','');
					right_proxy_content = right_proxy_content.replace('- ','');
					// if (annotation_uri=='my:new_york')
					// {
					// 	console.log('proxy_content:', proxy_content)
					// 	console.log('content:', content)
					// }
				}
				// const proxy_content = left_proxy_content+content+right_proxy_content;

				var left_missing_content = tokenise(left_proxy_content).join(' ');
				var right_missing_content = tokenise(right_proxy_content).join(' ');

				const joined_tokenised_proxy_content = left_missing_content + joined_splitted_content + right_missing_content;
				if (!label_is_in_content && joined_tokenised_proxy_content.search(regexp) < 0)
					continue;

				const label_is_short = token_list.length == 1;
				var left_missing_content = left_missing_content.trim();
				var right_missing_content = right_missing_content.trim();
				for (var s=splitted_content.length-1; !added && s>=0; --s)
				{
					// if (annotation_uri=='my:michael')
					// 	console.log('splitted_content[s]:', splitted_content, splitted_content[s], s)
					if (splitted_content[s] == '')
						continue
					var found = (splitted_content[s] == cleaned_sub_text);
					const last_s = s;

					if (!found && !label_is_short)
					{
						const min_s = s-token_list.length
						for (; s>=min_s && cleaned_sub_text.includes(splitted_content[s]) && !found; s--)
						{
							const content_portion = splitted_content.slice(s,last_s+1).join(' ');
							found = (content_portion == cleaned_sub_text);
							if (merge_span && !found)
							{
								var cannot_end = false;
								const is_at_end_of_content = (s == splitted_content.length-1);
								if (is_at_end_of_content)
								{
									const annotation_label_starts_with_content_portion = cleaned_sub_text.startsWith(content_portion);
									if (annotation_label_starts_with_content_portion)
									{
										const missing_end_of_content = cleaned_sub_text.slice(content_portion.length).trim();
										cannot_end = right_missing_content.startsWith(missing_end_of_content+' ') || (right_missing_content.length==missing_end_of_content.length && right_missing_content.startsWith(missing_end_of_content));
									}
								}
								// if (annotation_uri=='my:michael')
								// {
								// 	// console.log('content:', content)
								// 	console.log('	cannot_end:', cannot_end)
								// 	console.log('		is_at_end_of_content:', is_at_end_of_content, s)
								// 	console.log('		annotation_label_starts_with_content_portion:', cleaned_sub_text.startsWith(content_portion))
								// 	console.log('		left_missing_content:', left_missing_content)
								// 	console.log('		right_missing_content:', right_missing_content)
								// 	console.log('		missing_end_of_content:', cleaned_sub_text.slice(content_portion.length).trim())
								// 	console.log('		content_portion:', content_portion)
								// 	console.log('		cleaned_sub_text:', cleaned_sub_text)
								// }
								
								var cannot_start = false;
								const is_at_beginning_of_content = (s == 0);
								if (is_at_beginning_of_content)
								{
									const annotation_label_ends_with_content_portion = cleaned_sub_text.endsWith(content_portion);
									if (annotation_label_ends_with_content_portion)
									{
										const missing_beginning_of_content = cleaned_sub_text.slice(0,cleaned_sub_text.length-content_portion.length).trim();
										cannot_start = left_missing_content.endsWith(' '+missing_beginning_of_content) || (left_missing_content.length==missing_beginning_of_content.length && left_missing_content.endsWith(missing_beginning_of_content));
									}
								}
								
								// if (annotation_uri=='my:michael')
								// {
								// 	// console.log('content:', content)
								// 	console.log('	cannot_start:', cannot_start)
								// 	console.log('		is_at_beginning_of_content:', is_at_beginning_of_content, s)
								// 	console.log('		annotation_label_ends_with_content_portion:', cleaned_sub_text.endsWith(content_portion))
								// 	console.log('		left_missing_content:', left_missing_content)
								// 	console.log('		right_missing_content:', right_missing_content)
								// 	console.log('		missing_beginning_of_content:', cleaned_sub_text.slice(0,cleaned_sub_text.length-content_portion.length).trim())
								// 	console.log('		content_portion:', content_portion)
								// 	console.log('		cleaned_sub_text:', cleaned_sub_text)
								// }

								found = (cannot_start || cannot_end);
							}
							if (found)
								++s;
						}
					}

					if (found)
					{
						const start = [
							start_idx, // initial idx
							1 + s, // white spaces
							splitted_content.slice(0,s).map(x=>x.length).reduce((a,b)=>a+b, 0), // chars
						].reduce((a,b)=>a+b, 0);
						const end = [
							start, // idx of start
							last_s - s, // white spaces
							splitted_content.slice(s,last_s+1).map(x=>x.length).reduce((a,b)=>a+b, 0), // chars
						].reduce((a,b)=>a+b, 0);
						// keep the longest annotation, avoid nesting
						const initial_part = annotated_html.slice(0,start);
						const opening_annotation_tags = [...initial_part.matchAll(new RegExp('<annotation', 'gi'))];
						const closing_annotation_tags = [...initial_part.matchAll(new RegExp('</annotation>', 'gi'))];
						if (opening_annotation_tags.length == closing_annotation_tags.length)
						{
							const middle_part = annotated_html.slice(start,end);
							const final_part = annotated_html.slice(end);
							annotated_html = initial_part+annotate_fn(annotation_uri,middle_part,annotation_dict.text)+final_part;
						}
						added = true;
						start_idx = start;
					}
				}
			}
		}
		// console.log(start_idx,annotated_html.length);
	}
	return annotated_html;
}

function get_annotation_list_from_formatted_jsonld(jsonld)
{
	var sentenceListList = get_value_in_jsonld_by_key(jsonld, 'my:sentenceList');
	if (sentenceListList.length < 1)
		return [];
	var annotation_list = [];
	for (var sentenceList of sentenceListList)
	{
		for (var sentence of sentenceList)
		{
			if (ANNOTATION_LIST_URI in sentence)
			{
				annotation_list.push({
					'text': get_RDFItem_description(sentence[TEXT_URI]),
					'annotation': get_RDFItem_description(sentence[ANNOTATION_LIST_URI][0][RELATED_TO_URI]),
				});
			}
			if (WORD_ANNOTATION_LIST_URI in sentence)
			{
				for (var word_annotation of sentence[WORD_ANNOTATION_LIST_URI])
				{
					annotation_list.push({
						'text': get_RDFItem_description(word_annotation[TEXT_URI]),
						'annotation': get_RDFItem_description(word_annotation[ANNOTATION_LIST_URI][0][RELATED_TO_URI]),
					});
				}
			}
		}
	}
	return annotation_list
}

function get_process_input_dict_from_formatted_jsonld(jsonld)
{
	var process_list = get_unique_elements([].concat(...get_value_in_jsonld_by_key(jsonld, 'my:processList')), x=>get_dict_description(x));
	var process_input_dict = {};
	for (var p of process_list)
	{
		var process_input = p[PROCESS_INPUT_URI];
		var process_input_list = []
		for (var pi of process_input)
		{
			var input_dict = {}
			for (var k of ['@id', VALUE_URI, FEATURE_ORDER_URI, COUNTERFACTUAL_API_URI])
				input_dict[k] = get_RDFItem_description(pi[k]);
			process_input_list.push(input_dict);
		}
		process_input_dict[get_dict_description(p)] = process_input_list;
	}
	return process_input_dict;
}

function get_DOM_element_distance(element1,element2)
{
	var o1 = $(element1).offset();
	var o2 = $(element2).offset();
	var dx = o1.left - o2.left;
	var dy = o1.top - o2.top;
	return Math.sqrt(dx * dx + dy * dy);
}

function query_wikipedia_by_title(title, callback_fn)
{
	console.log(`Expanding ${title} on Wikipedia.`);
	try {
		$.ajax({
			url: "//en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro&explaintext&redirects=1&titles="+title, 
			method: 'GET',
			crossDomain: true,
			dataType: 'jsonp',
			success: function(data) {
				if (data.query && data.query.pages) 
				{
					for (var i in data.query.pages) 
					{
						var response = {
							'@id': title,
							'rdfs:label': data.query.pages[i].title,
							'dbo:abstract': data.query.pages[i].extract
						};
					}
					callback_fn(response);
				}
			},
		});
	} catch(e) {
		if (DEBUG)
			console.error(e);
	}
}

function titlefy(s) {
	if (!s)
		return s;
	// if ((s.match(/\(/g) || []).length != (s.match(/\)/g) || []).length)
	// 	s += ')';
	return s[0].toUpperCase() + s.slice(1);
}

function get(dict, key, def=null)
{
	var v = dict[key];
	return v?v:def;
}

function format_percentage(v, decimals=2)
{
	return (v*100).toFixed(decimals).toString().replace('.'+'0'.repeat(decimals),'')+'%';
}

function clip(n, min,max)
{
	if (n < min)
		return min;
	else if (n > max)
		return max;
	return n;
}

function getCookie(cname) {
	let name = cname + "=";
	let decodedCookie = decodeURIComponent(document.cookie);
	let ca = decodedCookie.split(';');
	for(let i = 0; i <ca.length; i++) {
		let c = ca[i];
		while (c.charAt(0) == ' ') {
			c = c.substring(1);
		}
		if (c.indexOf(name) == 0) {
			return c.substring(name.length, c.length);
		}
	}
	return "";
}