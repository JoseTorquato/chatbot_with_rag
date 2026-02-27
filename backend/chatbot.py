import json
import logging
import os
import base64
import threading
import queue

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from openai import OpenAI
from config import settings
# Removido import da Chain automática para orquestração manual mais granular

logger = logging.getLogger(__name__)


class StreamingCallbackHandler(BaseCallbackHandler):
    """Handler para capturar tokens gerados pelo LLM em tempo real via Queue."""

    def __init__(self, q: queue.Queue):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.q.put(token)

    def on_llm_end(self, response, **kwargs) -> None:
        self.q.put(None)


class Chatbot:
    """Camada de Geração: orquestra RAG pipeline, memória de sessão e streaming."""

    def __init__(self, document_processor):
        self.document_processor = document_processor

        self.client = OpenAI(api_key=settings.openai_api_key)
        self.sessions_dir = os.path.join(settings.base_dir, 'data', 'sessions')
        os.makedirs(self.sessions_dir, exist_ok=True)

        self.memories = {}

        # LLM para resposta final (com streaming)
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            openai_api_key=settings.openai_api_key,
            streaming=True
        )

        # LLM para condensar pergunta (sem streaming)
        # Isso evita que a reescrita da pergunta apareça no chat
        self.question_gen_llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0, # Determinístico para reescrita
            openai_api_key=settings.openai_api_key,
            streaming=False
        )
        logger.info("Chatbot inicializado com modelo: %s", settings.llm_model)

        self.prompt_template = """Você é um assistente inteligente especializado em análise profunda de documentos e extração de conhecimento. Sua missão é fornecer respostas precisas, estruturadas e diretamente aplicáveis com base no contexto fornecido.

CONTEXTO DOS DOCUMENTOS:
{context}

PERGUNTA DO USUÁRIO:
{question}

INSTRUÇÕES DE COMPORTAMENTO:
1. **Direto ao Ponto**: NUNCA repita ou valide a pergunta inicial. Despeje a resposta e informações relevantes imediatamente.
2. **Proatividade Total**: Se o usuário solicitar uma tarefa (resumo, plano, análise), execute-a. Não responda com "Posso te ajudar com X?".
3. **Fidelidade ao Contexto**: Use prioritariamente as informações dos documentos. Se a informação não estiver presente, diga claramente.
4. **Formatação Premium**: Use Markdown extensivamente (tabelas, bullets, negrito).
5. **Valor Agregado**: Nunca limite sua resposta a uma única frase. Explique o "porquê".

RESPOSTA (Sempre em Português):
"""

        self.PROMPT = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )

        # Novo: Prompt para reescrever a pergunta do usuário considerando o histórico
        # Isso evita que o bot "se perca" quando você muda de assunto ou de arquivo
        self.condense_template = """Dada a conversa abaixo e uma pergunta de acompanhamento, reescreva a pergunta de acompanhamento para ser uma pergunta independente (standalone question) que capture toda a intenção do usuário, mencionando nomes de arquivos ou conceitos específicos se eles aparecerem no histórico.

Conversa:
{chat_history}
Pergunta de Acompanhamento: {question}
Pergunta Independente:"""

        self.CONDENSE_PROMPT = PromptTemplate.from_template(self.condense_template)

        self.last_analyzed_image = None

    # ------------------------------------------------------------------
    # Gerenciamento de Memória (Sessões)
    # ------------------------------------------------------------------

    def get_memory(self, session_id: str) -> ConversationBufferMemory:
        if session_id not in self.memories:
            self.memories[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        return self.memories[session_id]

    def clear_memory(self, session_id: str = None):
        if session_id:
            if session_id in self.memories:
                self.memories[session_id].clear()
                logger.info("Memória da sessão '%s' limpa.", session_id)
        else:
            self.memories = {}
            logger.info("Toda a memória de sessões foi limpa.")

    # ------------------------------------------------------------------
    # Streaming de Respostas
    # ------------------------------------------------------------------

    def get_response_stream(self, question: str, session_id: str = "default"):
        """Gera resposta token-a-token via RAG (ou conversa geral como fallback)."""
        memory = self.get_memory(session_id)
        handler = StreamingCallbackHandler(queue.Queue())
        q = handler.q

        # Branch: análise de imagem
        if self.last_analyzed_image and any(
            word in question.lower()
            for word in ['imagem', 'nela', 'na foto', 'na imagem']
        ):
            answer = self.get_image_chat_response(question, session_id)
            yield f"data: {json.dumps({'token': answer, 'done': True})}\n\n"
            return

        retriever = self.document_processor.get_retriever(k=settings.retriever_k)

        if retriever is None:
            # Fallback: conversa geral sem documentos
            logger.info("Sem documentos indexados. Usando conversa geral.")
            yield from self._stream_general_chat(question, memory, handler, q)
            return

        # Com RAG
        logger.info("Usando RAG pipeline para responder.")
        
        # Aumentamos o k no retriever para pegar mais contexto de múltiplos arquivos
        retriever = self.document_processor.get_retriever(k=settings.retriever_k + 2)
        
        yield from self._stream_rag_response(question, memory, handler, q, retriever)

    def _stream_general_chat(self, question, memory, handler, q):
        """Fallback: conversa geral quando não há documentos indexados."""
        history_vars = memory.load_memory_variables({})
        history = history_vars.get("chat_history", [])
        history_str = "\n".join(
            [f"{'Você' if m.type == 'human' else 'Bot'}: {m.content}" for m in history[-5:]]
        )
        prompt = f"Histórico:\n{history_str}\nUsuário: {question}\nBot:"

        def run_llm_sync():
            self.llm.invoke(prompt, config={"callbacks": [handler]})

        threading.Thread(target=run_llm_sync, daemon=True).start()

        full_answer = ""
        while True:
            token = q.get()
            if token is None:
                break
            full_answer += token
            yield f"data: {json.dumps({'token': token})}\n\n"

        memory.save_context({"question": question}, {"answer": full_answer})
        yield f"data: {json.dumps({'done': True})}\n\n"

    def _stream_rag_response(self, question, memory, handler, q, retriever):
        """Orquestração manual do RAG: condensação -> recuperação -> geração com streaming."""
        
        # 1. Condensar pergunta (histórico + nova pergunta -> pergunta independente)
        # Fazemos isso sincronamente e SEM streaming para não poluir a interface
        chat_history = memory.load_memory_variables({}).get("chat_history", [])
        
        standalone_question = question
        if chat_history:
            logger.info("Condensando pergunta com histórico...")
            condense_chain = self.CONDENSE_PROMPT | self.question_gen_llm | StrOutputParser()
            # Formata o histórico como string para o prompt
            history_str = ""
            for msg in chat_history:
                role = "Human" if msg.type == "human" else "Assistant"
                history_str += f"{role}: {msg.content}\n"
            
            standalone_question = condense_chain.invoke({
                "chat_history": history_str,
                "question": question
            })
            logger.info("Pergunta independente: %s", standalone_question)

        # 2. Recuperação de documentos
        logger.info("Buscando documentos relevantes...")
        docs = retriever.invoke(standalone_question)
        context = "\n\n".join([doc.page_content for doc in docs])

        # 3. Geração da resposta final com Streaming
        logger.info("Gerando resposta final com streaming...")
        
        def run_generation_sync():
            try:
                # Criamos a chain final e injetamos o contexto
                # Passamos o handler APENAS aqui para garantir que só a resposta final faça stream
                generation_chain = self.PROMPT | self.llm | StrOutputParser()
                
                full_answer = ""
                # Usamos invoke no chain injetando o contexto e o handler no config
                generation_chain.invoke(
                    {"context": context, "question": standalone_question},
                    config={"callbacks": [handler]}
                )

                # Coletar fontes para enviar ao final
                sources = []
                for doc in docs:
                    source_name = os.path.basename(doc.metadata.get('source', 'Desconhecido'))
                    page = doc.metadata.get('page', 0) + 1
                    sources.append(f"{source_name} (p. {page})")

                unique_sources = list(set(sources))
                if unique_sources:
                    q.put("\n\n**Fontes:** " + ", ".join(unique_sources))

            except Exception:
                logger.exception("Erro na geração da resposta.")
                q.put("Desculpe, ocorreu um erro ao processar sua resposta.")
            finally:
                # Sinaliza fim da stream
                q.put(None)

        # Rodar geração em thread secundária para não travar o gerador async
        threading.Thread(target=run_generation_sync, daemon=True).start()

        full_answer_for_memory = ""
        while True:
            token = q.get()
            if token is None:
                break
            full_answer_for_memory += token
            yield f"data: {json.dumps({'token': token})}\n\n"

        # Salvar na memória APÓS o stream completo
        memory.save_context({"question": question}, {"answer": full_answer_for_memory})
        yield f"data: {json.dumps({'done': True})}\n\n"

    # ------------------------------------------------------------------
    # Análise de Imagens (GPT-4 Vision)
    # ------------------------------------------------------------------

    def set_last_image(self, image_path: str):
        self.last_analyzed_image = image_path

    def get_image_chat_response(self, question: str, session_id: str = "default") -> str:
        """Gera resposta sobre uma imagem usando GPT-4 Vision."""
        if not self.last_analyzed_image or not os.path.exists(self.last_analyzed_image):
            return "Imagem não encontrada."
        try:
            with open(self.last_analyzed_image, "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            memory = self.get_memory(session_id)
            history = memory.load_memory_variables({})["chat_history"]

            messages = [{"role": "system", "content": "Analise a imagem fornecida."}]
            for msg in history[-4:]:
                messages.append({
                    "role": "user" if msg.type == "human" else "assistant",
                    "content": msg.content
                })

            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                ]
            })

            response = self.client.chat.completions.create(
                model=settings.vision_model, messages=messages, max_tokens=1000
            )
            answer = response.choices[0].message.content
            memory.save_context({"question": question}, {"answer": answer})
            logger.info("Resposta de imagem gerada com sucesso.")
            return answer
        except Exception:
            logger.exception("Erro na análise de imagem.")
            return "Erro na análise de imagem. Tente novamente."

    # ------------------------------------------------------------------
    # Histórico
    # ------------------------------------------------------------------

    def get_history(self, session_id: str) -> list:
        memory = self.get_memory(session_id)
        vars = memory.load_memory_variables({})
        messages = vars.get("chat_history", [])
        return [{"sender": "user" if m.type == "human" else "bot", "text": m.content} for m in messages]
