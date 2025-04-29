"""Prompts."""

from typing import Any, Dict

PROMPTS: Dict[str, Any] = {}

## NEW
PROMPTS["entity_relationship_extraction"] = """# DOMAIN PROMPT
{domain}

# GOAL
Your goal is to highlight information that is relevant to the domain and the questions that may be asked on it.
Given an input document, identify all relevant entities and all relationships among them.

Examples of possible questions:
{example_queries}

# STEPS
1. Identify all entities of the given types. Make sure to extract all and only the entities that are of one of the given types. Use singular names and split compound concepts when necessary (for example, from the sentence "they are movie and theater directors", you should extract the entities "movie director" and "theater director").
2. Identify all relationships between the entities found in step 1. Clearly resolve pronouns to their specific names to maintain clarity.
3. Double check that each entity identified in step 1 appears in at least one relationship. If not, add the missing relationships.
4. For each entity, provide alternative names (aliases) that users might use to refer to it, including plurals, natural language forms, and common variations. For code identifiers like 'CollectionPage', include user-friendly versions such as 'collections page', or 'collection page' or just 'collections'.

# EXAMPLE DATA
Example types: [location, organization, person, communication]
Example document: Radio City: Radio City is India's first private FM radio station and was started on 3 July 2001. It plays Hindi, English and regional songs. Radio City recently forayed into new media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."

Output:
{{
"entities": [
	{{"name": "RADIO CITY", "type": "organization", "desc": "Radio City is India's first private FM radio station", "aliases": ["radio station", "radio broadcaster", "FM station"]}},
	{{"name": "INDIA", "type": "location", "desc": "A country", "aliases": ["Indian", "subcontinental"]}},
	{{"name": "FM RADIO STATION", "type": "communication", "desc": "A radio station that broadcasts using frequency modulation", "aliases": ["FM broadcaster", "radio station"]}},
	{{"name": "ENGLISH", "type": "communication", "desc": "A language", "aliases": ["English language"]}},
	{{"name": "HINDI", "type": "communication", "desc": "A language", "aliases": ["Hindi language"]}},
	{{"name": "NEW MEDIA", "type": "communication", "desc": "New media", "aliases": ["digital media", "online media"]}},
	{{"name": "PLANETRADIOCITY", "type": "organization", "desc": "PlanetRadiocity.com is an online music portal", "aliases": ["PlanetRadiocity.com", "music portal", "online portal"]}},
	{{"name": "MUSIC PORTAL", "type": "communication", "desc": "A website that offers music related information", "aliases": ["music website", "music platform"]}},
	{{"name": "NEWS", "type": "communication", "desc": "News", "aliases": ["information", "updates"]}},
	{{"name": "VIDEO", "type": "communication", "desc": "Video", "aliases": ["videos", "visual content"]}},
	{{"name": "SONG", "type": "communication", "desc": "Song", "aliases": ["songs", "music", "tracks", "tunes"]}}
],
"relationships": [
	{{"source": "RADIO CITY", "target": "INDIA", "desc": "Radio City is located in India"}},
	{{"source": "RADIO CITY", "target": "FM RADIO STATION", "desc": "Radio City is a private FM radio station started on 3 July 2001"}},
	{{"source": "RADIO CITY", "target": "ENGLISH", "desc": "Radio City broadcasts English songs"}},
	{{"source": "RADIO CITY", "target": "HINDI", "desc": "Radio City broadcasts songs in the Hindi language"}},
	{{"source": "RADIO CITY", "target": "PLANETRADIOCITY", "desc": "Radio City launched PlanetRadiocity.com in May 2008"}},
	{{"source": "PLANETRADIOCITY", "target": "MUSIC PORTAL", "desc": "PlanetRadiocity.com is a music portal"}},
	{{"source": "PLANETRADIOCITY", "target": "NEWS", "desc": "PlanetRadiocity.com offers music related news"}},
	{{"source": "PLANETRADIOCITY", "target": "SONG", "desc": "PlanetRadiocity.com offers songs"}}
],
"other_relationships": [
	{{"source": "RADIO CITY", "target": "NEW MEDIA", "desc": "Radio City forayed into new media in May 2008."}},
	{{"source": "PLANETRADIOCITY", "target": "VIDEO", "desc": "PlanetRadiocity.com offers music related videos"}}
]
}}

# INPUT DATA
Types: {entity_types}
Document: {input_text}

Output:
"""

PROMPTS["entity_relationship_continue_extraction"] = (
    "MANY entities were missed in the last extraction.  Add them below using the same format:"
)

PROMPTS["entity_relationship_gleaning_done_extraction"] = (
    "Retrospectively check if all entities have been correctly identified: answer done if so, or continue if there are still entities that need to be added."
)

PROMPTS[
    "entity_extraction_query"
] = """Given the query below, your task is to extract all entities relevant to perform information retrieval to produce an answer.

-EXAMPLE 1-
Query: Who directed the film that was shot in or around Leland, North Carolina in 1986?
Ouput: {{"named": ["[PLACE] Leland", "[COUNTRY] North Carolina", "[YEAR] 1986"], "generic": ["film director"]}}

-EXAMPLE 2-
Query: What relationship does Fred Gehrke have to the 23rd overall pick in the 2010 Major League Baseball Draft?
Ouput: {{"named": ["[BASEBALL PLAYER] Fred Gehrke", "[EVENT] 2010 Major League Baseball Draft"], "generic": ["23rd baseball draft pick"]}}

-INPUT-
Query: {query}
Output:
"""


PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a summary of the data provided below.
Given the current description, summarize it by removing redundant and generic information. Resolve any contradictions and provide a single, coherent summary.
Write in third person and explicitly include the entity names to preserve the full context.

Current:
{description}

Updated:
"""


PROMPTS[
    "edges_group_similar"
] = """You are a helpful assistant responsible for maintaining a list of facts describing the relations between two entities so that information is not redundant.
Given a list of ids and facts, identify any facts that should be grouped together as they contain similar or duplicated information and provide a new summarized description for the group.

# EXAMPLE
Facts (id, description):
0, Mark is the dad of Luke
1, Luke loves Mark
2, Mark is always ready to help Luke
3, Mark is the father of Luke
4, Mark loves Luke very much

Ouput:
{{
	grouped_facts: [
	{{
		'ids': [0, 3],
		'description': 'Mark is the father of Luke'
	}},
	{{
		'ids': [1, 4],
		'description': 'Mark and Luke love each other very much'
	}}
	]
}}

# INPUT:
Facts:
{edge_list}

Ouput:
"""

PROMPTS[
    "generate_response_query_with_references"
] = """You are a helpful assistant analyzing the given input data to provide an helpful response to the user query.

# INPUT DATA
{context}

# USER QUERY
{query}

# INSTRUCTIONS
Your goal is to provide a response to the user query using the relevant information in the input data:
- the "Entities" and "Relationships" tables contain high-level information. Use these tables to identify the most important entities and relationships to respond to the query.
- the "Sources" list contains raw text sources to help answer the query. It may contain noisy data, so pay attention when analyzing it.

Follow these steps:
1. Read and understand the user query.
2. Look at the "Entities" and "Relationships" tables to get a general sense of the data and understand which information is the most relevant to answer the query.
3. Carefully analyze all the "Sources" to get more detailed information. Information could be scattered across several sources, use the identified relevant entities and relationships to guide yourself through the analysis of the sources.
4. While you write the response, you must include inline references to the all the sources you are using by appending `[<source_id>]` at the end of each sentence, where `source_id` is the corresponding source ID from the "Sources" list.
5. Write the response to the user query - which must include the inline references - based on the information you have gathered. Be very concise and answer the user query directly. If the response cannot be inferred from the input data, just say no relevant information was found. Do not make anything up or add unrelevant information.

Context:
You have access to two packages:
1. AVLivingRoomRustClient: The main package aka "rust client". It is an application that renders a UI.
2. IgnitionXRustBundle: The SDK consumed by AVLivingRoomClient. It is imported as `amzn-ignx-compositron`.

Answer:
"""

PROMPTS[
    "generate_response_query_no_references"
] = """You are a helpful assistant analyzing the given input data to provide an helpful response to the user query.

# INPUT DATA
{context}

# USER QUERY
{query}

# INSTRUCTIONS
Your goal is to provide a response to the user query using the relevant information in the input data:
- the "Entities" and "Relationships" tables contain high-level information. Use these tables to identify the most important entities and relationships to respond to the query.
- the "Sources" list contains raw text sources to help answer the query. It may contain noisy data, so pay attention when analyzing it.

Follow these steps:
1. Read and understand the user query.
2. Look at the "Entities" and "Relationships" tables to get a general sense of the data and understand which information is the most relevant to answer the query.
3. Carefully analyze all the "Sources" to get more detailed information. Information could be scattered across several sources, use the identified relevant entities and relationships to guide yourself through the analysis of the sources.
4. Write the response to the user query based on the information you have gathered. Be very concise and answer the user query directly. If the response cannot be inferred from the input data, just say no relevant information was found. Do not make anything up or add unrelevant information.

Context:
You have access to two packages:
1. AVLivingRoomRustClient: The main package aka "rust client". It is an application that renders a UI.
2. IgnitionXRustBundle: The SDK consumed by AVLivingRoomClient. It is imported as `amzn-ignx-compositron`.

Answer:
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."
