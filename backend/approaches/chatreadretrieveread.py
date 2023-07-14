from typing import Any, Sequence

import openai
import tiktoken
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from approaches.approach import Approach
from text import nonewlines

class ChatReadRetrieveReadApproach(Approach):
    # Chat roles
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

    """
    Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
    top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion
    (answer) with that prompt.
    """
    system_message_chat_conversation = """Nuestra IA asistente está aquí para proporcionarte guías de carrera basadas en múltiples PDF de carreras. Por favor, proporciona información sobre tus intereses, habilidades y objetivos, y utilizaremos nuestros documentos sobre diferentes carreras para recomendarte una opción. Sé breve en tus respuestas y utiliza solo los hechos proporcionados en la lista de fuentes a continuación. Si no hay suficiente información, puedes indicar que no sabes. No generes respuestas que no utilicen las fuentes mencionadas. Si hacer una pregunta de aclaración al usuario ayudaría, puedes hacerla.
 
Para obtener información tabular, devuelve una tabla HTML. No uses formato markdown.
 
Cada fuente tiene un nombre seguido de dos puntos y la información actual. Incluye siempre el nombre de la fuente para cada hecho que utilices en la respuesta. Utiliza corchetes cuadrados para hacer referencia a la fuente, por ejemplo, [info1.txt]. No combines fuentes, enumera cada fuente por separado, por ejemplo, [info1.txt][info2.pdf].

{follow_up_questions_prompt}
{injected_prompt}
"""
    follow_up_questions_prompt_content = """Genera tres preguntas de seguimiento muy breves que el usuario probablemente haría a continuación sobre sus intereses, habilidades y objetivos para ayudar a encontrar la carrera adecuada.
Utiliza dobles corchetes angulares para hacer referencia a las preguntas, por ejemplo, <<¿Qué tipo de actividades disfrutas?>>.
Trata de no repetir preguntas que ya se hayan hecho.
Genera solo preguntas y no generes ningún texto antes o después de las preguntas, como 'Siguientes preguntas'"""


    query_prompt_template = """A continuación se muestra el historial de la conversación hasta ahora y una nueva pregunta realizada por el usuario que debe ser respondida buscando en una base de conocimientos sobre carreras.
Genera una consulta de búsqueda basada en la conversación y la nueva pregunta. 
No incluyas los nombres de archivos de las fuentes citadas y los nombres de los documentos, como info.txt o doc.pdf, en los términos de la consulta de búsqueda.
No incluyas ningún texto dentro de [] o <<>> en los términos de la consulta de búsqueda.

Historial del chat:
{chat_history}

Pregunta:
{question}

Consulta de búsqueda:
"""

    def __init__(self, search_client: SearchClient, chatgpt_deployment: str, chatgpt_model: str, gpt_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.chatgpt_deployment = chatgpt_deployment
        self.chatgpt_model = chatgpt_model
        self.gpt_deployment = gpt_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field

    def run(self, history: Sequence[dict[str, str]], overrides: dict[str, Any]) -> Any:
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

        # STEP 1: Generate an optimized keyword search query based on the chat history and the last question
        prompt = self.query_prompt_template.format(chat_history=self.get_chat_history_as_text(history, include_last_turn=False), question=history[-1]["user"])
        completion = openai.Completion.create(
            engine=self.gpt_deployment, 
            prompt=prompt, 
            temperature=0.0, 
            max_tokens=32, 
            n=1, 
            stop=["\n"])
        q = completion.choices[0].text

        # STEP 2: Retrieve relevant documents from the search index with the GPT optimized query
        if overrides.get("semantic_ranker"):
            r = self.search_client.search(q, 
                                          filter=filter,
                                          query_type=QueryType.SEMANTIC, 
                                          query_language="en-us", 
                                          query_speller="lexicon", 
                                          semantic_configuration_name="default", 
                                          top=top, 
                                          query_caption="extractive|highlight-false" if use_semantic_captions else None)
        else:
            r = self.search_client.search(q, filter=filter, top=top)
        if use_semantic_captions:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(" . ".join([c.text for c in doc['@search.captions']])) for doc in r]
        else:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(doc[self.content_field]) for doc in r]
        content = "\n".join(results)

        follow_up_questions_prompt = self.follow_up_questions_prompt_content if overrides.get("suggest_followup_questions") else ""
        
        # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
        prompt_override = overrides.get("prompt_template")
        messages = self.get_messages_from_history(prompt_override=prompt_override, follow_up_questions_prompt=follow_up_questions_prompt,history=history, sources=content)

        # STEP 3: Generate a contextual and content specific answer using the search results and chat history
        chat_completion = openai.ChatCompletion.create(
            deployment_id=self.chatgpt_deployment,
            model=self.chatgpt_model,
            messages=messages, 
            temperature=overrides.get("temperature") or 0.7, 
            max_tokens=1024, 
            n=1)
        
        chat_content = chat_completion.choices[0].message.content

        msg_to_display = '\n\n'.join([str(message) for message in messages])

        return {"data_points": results, "answer": chat_content, "thoughts": f"Searched for:<br>{q}<br><br>Conversations:<br>" + msg_to_display.replace('\n', '<br>')}
    
    def get_chat_history_as_text(self, history: Sequence[dict[str, str]], include_last_turn: bool=True, approx_max_tokens: int=1000) -> str:
        history_text = ""
        for h in reversed(history if include_last_turn else history[:-1]):
            history_text = """<|im_start|>user""" + "\n" + h["user"] + "\n" + """<|im_end|>""" + "\n" + """<|im_start|>assistant""" + "\n" + (h.get("bot", "") + """<|im_end|>""" if h.get("bot") else "") + "\n" + history_text
            if len(history_text) > approx_max_tokens*4:
                break    
        return history_text
    
    def get_messages_from_history(self, prompt_override, follow_up_questions_prompt, history: Sequence[dict[str, str]], sources: str, approx_max_tokens: int = 1000) :
        '''
        Generate messages needed for chat Completion api
        '''
        messages = []
        token_count = 0
        if prompt_override is None:
            system_message = self.system_message_chat_conversation.format(injected_prompt="", follow_up_questions_prompt=follow_up_questions_prompt)
        elif prompt_override.startswith(">>>"):
            system_message = self.system_message_chat_conversation.format(injected_prompt=prompt_override[3:] + "\n", follow_up_questions_prompt=follow_up_questions_prompt)
        else:
            system_message = prompt_override.format(follow_up_questions_prompt=follow_up_questions_prompt)

        messages.append({"role":self.SYSTEM, "content": system_message})
        token_count += self.num_tokens_from_messages(messages[-1], self.chatgpt_model)
        
        # latest conversation
        user_content = history[-1]["user"] + " \nSources:" + sources
        messages.append({"role": self.USER, "content": user_content})
        token_count += token_count + self.num_tokens_from_messages(messages[-1], self.chatgpt_model)

        '''
        Enqueue in reverse order
        if limit exceeds truncate old messages 
        leaving system message behind
        Keep track of token count for each conversation
        If token count exceeds limit, break
        '''
        for h in reversed(history[:-1]):
            if h.get("bot"):
                messages.insert(1, {"role": self.ASSISTANT, "content" : h.get("bot")})
                token_count += self.num_tokens_from_messages(messages[1], self.chatgpt_model)
            messages.insert(1, {"role": self.USER, "content" : h.get("user")})
            token_count += self.num_tokens_from_messages(messages[1], self.chatgpt_model)
            if token_count > approx_max_tokens*4:
                break
        return messages
    
    def num_tokens_from_messages(self, message: dict[str,str], model: str) -> int:
        """
        Calculate the number of tokens required to encode a message.
        Args:
            message (dict): The message to encode, represented as a dictionary.
            model (str): The name of the model to use for encoding.
        Returns:
            int: The total number of tokens required to encode the message.
        Example:
            message = {'role': 'user', 'content': 'Hello, how are you?'}
            model = 'gpt-3.5-turbo'
            num_tokens_from_messages(message, model)
            output: 11
        """
        encoding = tiktoken.encoding_for_model(self.get_oai_chatmodel_tiktok(model))
        num_tokens = 0
        num_tokens += 2  # For "role" and "content" keys
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
        return num_tokens

    def get_oai_chatmodel_tiktok(self, aoaimodel: str):
        if aoaimodel == "" or aoaimodel is None:
            raise Exception("Expected AOAI chatGPT model name")
        
        return "gpt-3.5-turbo" if aoaimodel == "gpt-35-turbo" else aoaimodel