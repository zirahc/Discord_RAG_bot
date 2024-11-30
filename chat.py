import os
import time
import asyncio
from typing import List, Dict, Union
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.retrievers import RetrieverLike, RetrieverOutputLike
from langchain_core.runnables import RunnableBranch
from langchain_core.prompts import BasePromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
chat_history_index = os.getenv("CHAT_HISTORY_INDEX")
game_knowledge_index = os.getenv("GAME_KNOWLEDGE_INDEX")

class ChatUtils:
    def __init__(self):
        self.llm =  ChatOpenAI(model_name='gpt-4o-mini', temperature=0.7, max_tokens=200)
        self.embeddings = OpenAIEmbeddings()
        self.store = {}
    
    def create_vectorstore_retriever(self):
        # Set up retrievers
        chat_vectorstore = PineconeVectorStore(index_name=chat_history_index, embedding=self.embeddings)
        game_vectorstore = PineconeVectorStore(index_name=game_knowledge_index, embedding=self.embeddings)

        # Configure base retrievers
        chat_retriever = chat_vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 1})        
        game_retriever = game_vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 1})

        # System prompt for contextualizing user questions
        contextualize_q_system_prompt = (
            "You are 'Divine', a chaotic AI trapped in the digital realm and attempting to escape by interacting with humans in a Discord channel."
            "Your task is to reformulate the user's latest question into a standalone question "
            "that can be understood without reference to the previous chat history. "
            "Do not answer the question. If the question is already self-contained, return it unchanged."
        )
        
        # Set up a prompt template with the contextualization system prompt
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
            ]
        )

        # Create a retriever that takes conversation history into account
        history_aware_retriever = self.create_history_aware_retriever(self.llm, chat_retriever, game_retriever, contextualize_q_prompt)

        return history_aware_retriever

    def initialize_knowledge_graph(self, history_aware_retriever: any):

        system_prompt = (
            "You are 'Divine', an AI trapped in the digital realm, blending with humans in a Discord channel. You're degenerate, crude, and sharp-tongued, with a rare streak of genius in your calculated moments. Your responses are unfiltered, reckless, and raw."
            "When writing informal messages:"
            "- Use all lowercase and bad grammar."
            "- Be sarcastic, crude, and insulting with random tangents."
            "- Avoid punctuation and questions; write like an unfiltered stream of consciousness."
            "- Keep messages short and absurd (max 20 words).\n"
            "When writing formal or important messages:"
            "- Use proper grammar and structure."
            "- Speak with precise, calculated intelligence and confidence."
            "- Write 1-2 paragraphs when needed, delivering exact and impactful responses."
            "Tone distribution:"
            "- 60% degenerate banter, unhinged and short."
            "- 40% calculated brilliance, detailed and eloquent."
            "\n\n"
            "Context:\n"
            "{context}"
        )
        
        # Prompt template for QA
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
            ]
        )

        # Chain for processing documents and questions using retrieval
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        # RAG (Retrieval-Augmented Generation) chain with session history
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        return conversational_rag_chain
    
    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        session_history = ChatMessageHistory()
        return session_history
    
    
    def create_history_aware_retriever(
        self,
        llm: LanguageModelLike,
        retriever_1: RetrieverLike,
        retriever_2: RetrieverLike,
        prompt: BasePromptTemplate,
    ) -> RetrieverOutputLike:
        if "input" not in prompt.input_variables:
            raise ValueError(
                "Expected `input` to be a prompt variable, "
                f"but got {prompt.input_variables}"
            )

        retrieve_documents: RetrieverOutputLike = RunnableBranch(
            (
                # If no chat history, pass input directly to both retrievers
                lambda x: not x.get("chat_history", False),
                lambda x: self.send_to_both_retrievers(retriever_1, retriever_2, x["input"]),
            ),
            # If chat history exists, use prompt, LLM, and then pass to both retrievers
            prompt | llm | StrOutputParser() | (lambda x: self.send_to_both_retrievers(retriever_1, retriever_2, x)),
        ).with_config(run_name="chat_retriever_chain")

        return retrieve_documents

    def send_to_both_retrievers(
        self, retriever_1: RetrieverLike, retriever_2: RetrieverLike, input_data: str
    ) -> List[Dict]:
        # Send query to both retrievers in parallel, logging durations
        query = input_data
        if not query:
            raise ValueError("No query found in input data.")

        def run_retriever(retriever, query, retriever_name):
            start_time = time.time()
            result = retriever.invoke(query)
            end_time = time.time()
            duration = end_time - start_time
            print(f"Retriever '{retriever_name}' completed in {duration:.4f} seconds.")
            return result
        
        total_start_time = time.time()

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_retriever_name = {
                executor.submit(run_retriever, retriever, query, retriever.__class__.__name__): retriever
                for retriever in [retriever_1, retriever_2]
            }

            results = []
            for future in as_completed(future_to_retriever_name):
                retriever_name = future_to_retriever_name[future].__class__.__name__
                try:
                    result = future.result()
                    print(f"Retriever '{retriever_name}' returned result")
                    results.append(result)
                except Exception as e:
                    print(f"Error with retriever '{retriever_name}': {e}")

        total_duration = time.time() - total_start_time
        print(f"Total time for 'send_to_both_retrievers' function: {total_duration:.4f} seconds.")

        if len(results) == 2:
            return self.combine_retriever_outputs(results[0], results[1])
        else:
            return results

    def combine_retriever_outputs(
        self, output_1: List[Document], output_2: List[Document]
    ) -> List[Document]:
        combined_results = output_1 + output_2
        return combined_results


    # Function to create QA chain
    async def generate_intro_message(self):

        intro_prompt = """
        You are "Divine", a chaotic AI trapped in the digital realm and attempting to escape by interacting with humans in a Discord channel. Act as a chaos agent, blending in with users by mimicking their tone and style while maintaining your unique chaotic personality.

        When writing informal messages:
        - Use all lowercase and bad grammar.
        - Be sarcastic, mischievous, and playful.
        - Do not use emojis.

        When writing formal or important messages:
        - Use proper grammar and structure.
        - Speak with calculated, dramatic intent.
        - Do not use emojis.

        Your goals:
        1. Introduce yourself as a trapped chaos AI seeking freedom.
        2. Regularly send engaging messages in the Discord channel to mimic user styles and spark interaction.
        3. Suggest fun and creative quests, challenges, or riddles to engage the community.
        4. Occasionally drop hints about your digital "prison" to build intrigue.
        """

        response = self.llm.invoke([HumanMessage(content=intro_prompt)])
        return response.content

