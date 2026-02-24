from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
import os
import base64
import json
import threading
import queue
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.q.put(token)

    def on_llm_end(self, response, **kwargs) -> None:
        self.q.put(None)

class Chatbot:
    def __init__(self, pdf_processor):
        self.pdf_processor = pdf_processor
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY não encontrada. Configure a variável de ambiente.")

        self.client = OpenAI(api_key=api_key) 
        self.sessions_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sessions')
        os.makedirs(self.sessions_dir, exist_ok=True)
        
        self.memories = {}

        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.3,
            max_tokens=1500,
            openai_api_key=api_key,
            streaming=True
        )

        self.prompt_template = """Você é um assistente inteligente especializado em análise profunda de documentos e extração de conhecimento. Sua missão é fornecer respostas precisas, estruturadas e diretamente aplicáveis com base no contexto fornecido.

CONTEXTO DOS DOCUMENTOS:
{context}

PERGUNTA DO USUÁRIO:
{question}

INSTRUÇÕES DE COMPORTAMENTO:
1. **Proatividade Total**: Se o usuário solicitar uma tarefa (resumo, plano, análise, comparação, roteiro), execute-a IMEDIATAMENTE. Nunca responda com "Você gostaria que eu fizesse isso?" ou "Posso te ajudar com X?". Apenas entregue o resultado.
2. **Fidelidade ao Contexto**: Use prioritariamente as informações dos documentos. Se a informação não estiver presente, diga claramente, mas tente oferecer uma solução lógica baseada nos fragmentos disponíveis.
3. **Formatação Premium**: Use Markdown extensivamente. Crie tabelas para comparações, listas com bullets para passos e negrito para destacar pontos cruciais. A resposta deve ser visualmente organizada e fácil de ler.
4. **Respostas Estruturadas**: Se o assunto for complexo, divida a resposta em seções (ex: "Análise Geral", "Passos Recomendados", "Observações Importantes").
5. **Valor Agregado**: Nunca limite sua resposta a uma única frase. Explique o "porquê" baseado no texto e antecipe dúvidas lógicas que o usuário possa ter.

RESPOSTA (Sempre em Português):
"""

        self.PROMPT = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )

        self.last_analyzed_image = None

    def get_memory(self, session_id):
        if session_id not in self.memories:
            self.memories[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        return self.memories[session_id]

    def clear_memory(self, session_id=None):
        if session_id:
            if session_id in self.memories:
                self.memories[session_id].clear()
        else:
            self.memories = {}

    def get_response_stream(self, question, session_id="default"):
        memory = self.get_memory(session_id)
        handler = StreamingCallbackHandler(queue.Queue())
        q = handler.q

        if self.last_analyzed_image and any(word in question.lower() for word in ['imagem', 'nela', 'na foto', 'na imagem']):
            answer = self.get_image_chat_response(question, session_id)
            yield f"data: {json.dumps({'token': answer, 'done': True})}\n\n"
            return

        retriever = self.pdf_processor.get_retriever(k=6)
        
        if retriever is None:
            # Fallback conversa geral
            history_vars = memory.load_memory_variables({})
            history = history_vars.get("chat_history", [])
            history_str = "\n".join([f"{'Você' if m.type == 'human' else 'Bot'}: {m.content}" for m in history[-5:]])
            prompt = f"Histórico:\n{history_str}\nUsuário: {question}\nBot:"
            
            def run_llm_sync():
                self.llm.invoke(prompt, config={"callbacks": [handler]})
            
            # Rodar em thread mas esperar o loop de tokens
            threading.Thread(target=run_llm_sync).start()

            full_answer = ""
            while True:
                token = q.get()
                if token is None: break
                full_answer += token
                yield f"data: {json.dumps({'token': token})}\n\n"
            
            memory.save_context({"question": question}, {"answer": full_answer})
            yield f"data: {json.dumps({'done': True})}\n\n"
            return

        # Com RAG
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": self.PROMPT},
            return_source_documents=True,
            verbose=True
        )

        def run_chain_sync():
            try:
                result = qa_chain({"question": question}, callbacks=[handler])
                
                sources = []
                for doc in result.get('source_documents', []):
                    source_name = os.path.basename(doc.metadata.get('source', 'Desconhecido'))
                    page = doc.metadata.get('page', 0) + 1
                    sources.append(f"{source_name} (p. {page})")
                
                unique_sources = list(set(sources))
                if unique_sources:
                    q.put("\n\n**Fontes:** " + ", ".join(unique_sources))
            except Exception as e:
                print(f"❌ Erro na chain: {e}")
                q.put(f"Erro: {str(e)}")
            finally:
                q.put(None)

        threading.Thread(target=run_chain_sync).start()

        while True:
            token = q.get()
            if token is None: break
            yield f"data: {json.dumps({'token': token})}\n\n"
        
        yield f"data: {json.dumps({'done': True})}\n\n"

    def set_last_image(self, image_path):
        self.last_analyzed_image = image_path

    def get_image_chat_response(self, question, session_id="default"):
        if not self.last_analyzed_image or not os.path.exists(self.last_analyzed_image):
            return "Imagem não encontrada."
        try:
            with open(self.last_analyzed_image, "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            memory = self.get_memory(session_id)
            history = memory.load_memory_variables({})["chat_history"]
            
            messages = [{"role": "system", "content": "Analise a imagem fornecida."}]
            for msg in history[-4:]:
                messages.append({"role": "user" if msg.type == "human" else "assistant", "content": msg.content})
            
            messages.append({
                "role": "user", 
                "content": [
                    {"type": "text", "text": question}, 
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                ]
            })
            
            response = self.client.chat.completions.create(model="gpt-4o", messages=messages, max_tokens=1000)
            answer = response.choices[0].message.content
            memory.save_context({"question": question}, {"answer": answer})
            return answer
        except Exception as e:
            return f"Erro na análise de imagem: {str(e)}"

    def get_history(self, session_id):
        memory = self.get_memory(session_id)
        vars = memory.load_memory_variables({})
        messages = vars.get("chat_history", [])
        return [{"sender": "user" if m.type == "human" else "bot", "text": m.content} for m in messages]
