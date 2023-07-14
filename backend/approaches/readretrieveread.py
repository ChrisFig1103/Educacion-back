import openai
from approaches.approach import Approach
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from langchain.llms.openai import AzureOpenAI
from langchain.callbacks.manager import CallbackManager, Callbacks
from langchain.chains import LLMChain
from langchain.agents import Tool, ZeroShotAgent, AgentExecutor
from langchainadapters import HtmlCallbackHandler
from text import nonewlines
from lookuptool import CsvLookupTool
from typing import Any

class ReadRetrieveReadApproach(Approach):
    """
    Attempt to answer questions by iteratively evaluating the question to see what information is missing, and once all information
    is present then formulate an answer. Each iteration consists of two parts:
     1. use GPT to see if we need more information
     2. if more data is needed, use the requested "tool" to retrieve it.
    The last call to GPT answers the actual question.
    This is inspired by the MKRL paper[1] and applied here using the implementation in Langchain.

    [1] E. Karpas, et al. arXiv:2205.00445
    """

    template_prefix = \
"You are an intelligent assistant helping Contoso Inc employees with their healthcare plan questions and employee handbook questions. " + \
"Use 'you' to refer to the individual asking the questions even if they ask with 'I'. " + \
"Answer the following question using only the data provided in the sources below. " + \
"For tabular information return it as an html table. Do not return markdown format. "  + \
"Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. " + \
"If you cannot answer using the sources below, say you don't know. " + \
"""

###
Question: 'Which career is better suited for someone interested in {interests}'?

Sources:
{retrieved}

Answer:
"""    

    CognitiveSearchToolDescription = "useful for searching the Microsoft employee benefits information such as healthcare plans, retirement plans, etc."

    def __init__(self, search_client: SearchClient, openai_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.openai_deployment = openai_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field

    def retrieve(self, q: str, overrides: dict[str, Any]) -> Any:
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

        if overrides.get("semantic_ranker"):
            r = self.search_client.search(q,
                                          filter=filter, 
                                          query_type=QueryType.SEMANTIC, 
                                          query_language="en-us", 
                                          query_speller="lexicon", 
                                          semantic_configuration_name="default", 
                                          top = top,
                                          query_caption="extractive|highlight-false" if use_semantic_captions else None)
        else:
            r = self.search_client.search(q, filter=filter, top=top)
        if use_semantic_captions:
            self.results = [doc[self.sourcepage_field] + ":" + nonewlines(" -.- ".join([c.text for c in doc['@search.captions']])) for doc in r]
        else:
            self.results = [doc[self.sourcepage_field] + ":" + nonewlines(doc[self.content_field][:250]) for doc in r]
        content = "\n".join(self.results)
        return content
        
    def run(self, q: str, overrides: dict[str, Any]) -> Any:
        # Not great to keep this as instance state, won't work with interleaving (e.g. if using async), but keeps the example simple
        self.results = None

        # Use to capture thought process during iterations
        cb_handler = HtmlCallbackHandler()
        cb_manager = CallbackManager(handlers=[cb_handler])
        
        acs_tool = Tool(name="CognitiveSearch", 
                        func=lambda q: self.retrieve(q, overrides), 
                        description=self.CognitiveSearchToolDescription,
                        callbacks=cb_manager)
        employee_tool = EmployeeInfoTool("Employee1", callbacks=cb_manager)
        tools = [acs_tool, employee_tool]

        prompt = ZeroShotAgent.create_prompt(
            tools=tools,
            prefix=overrides.get("prompt_template_prefix") or self.template_prefix,
            suffix=overrides.get("prompt_template_suffix") or self.template_suffix,
            input_variables = ["input", "agent_scratchpad"])
        llm = AzureOpenAI(deployment_name=self.openai_deployment, temperature=overrides.get("temperature") or 0.3, openai_api_key=openai.api_key)
        chain = LLMChain(llm = llm, prompt = prompt)
        agent_exec = AgentExecutor.from_agent_and_tools(
            agent = ZeroShotAgent(llm_chain = chain, tools = tools),
            tools = tools, 
            verbose = True, 
            callback_manager = cb_manager)
        result = agent_exec.run(q)
                
        # Remove references to tool names that might be confused with a citation
        result = result.replace("[CognitiveSearch]", "").replace("[Employee]", "")

        return {"data_points": self.results or [], "answer": result, "thoughts": cb_handler.get_and_reset_log()}

class EmployeeInfoTool(CsvLookupTool):
    employee_name: str = ""

    def __init__(self, employee_name: str, callbacks: Callbacks = None):
        super().__init__(filename="data/employeeinfo.csv", 
                         key_field="name", 
                         name="Employee", 
                         description="useful for answering questions about the employee, their benefits and other personal information",
                         callbacks=callbacks)
        self.func = self.employee_info
        self.employee_name = employee_name

    def employee_info(self, name: str) -> str:
        return self.lookup(name)
