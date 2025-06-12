from cpt_categorizer.agents.parsing import ParsingResult
from cpt_categorizer.agents.tagging import (
    tag_section_and_subsection,
    SECTION_CLASSIFIER_PROMPT,
)
from cpt_categorizer.config.openai import get_openai_client

parsing_result = ParsingResult(
    cleaned="EXTRACORPOREAL SHOCK WAVE LITHOTRIPSY (ESWL) (NET OF PHIC, PF INCLUDED),7042",
    confidence_score=0.95,
    is_ambiguous=False,
    error=None,
)

output = tag_section_and_subsection(parsing_result, client=get_openai_client())

print(output)
