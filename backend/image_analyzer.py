import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class ImageAnalyzer:
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY não encontrada. Configure a variável de ambiente.")
        
        self.client = OpenAI(api_key=api_key)
        
        self.prompt = """Você é um especialista em design, UI/UX e análise visual. Analise esta imagem cuidadosamente e forneça um feedback detalhado e construtivo.

**Seu feedback deve incluir:**

1. **Descrição Geral:** O que você vê na imagem?

2. **Pontos Fortes:** O que está funcionando bem?
   - Design/estética
   - Composição
   - Cores e contraste
   - Clareza da mensagem

3. **Pontos de Melhoria:** O que pode ser melhorado? Seja específico!
   - Problemas de design
   - Questões de usabilidade
   - Acessibilidade
   - Hierarquia visual
   - Tipografia
   - Espaçamento

4. **Sugestões Práticas:** Como implementar as melhorias?
   - Passos concretos
   - Exemplos específicos
   - Alternativas

5. **Conclusão:** Resumo final com prioridades.

**Importante:**
- Seja honesto mas construtivo
- Dê exemplos específicos
- Priorize as melhorias mais importantes
- Responda em português do Brasil
- Formate sua resposta de forma clara e organizada"""
    
    def analyze_image(self, image_path):
        print(f"\n🖼️ Analisando imagem: {image_path}")
        
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        file_extension = os.path.splitext(image_path)[1].lower()
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        mime_type = mime_types.get(file_extension, 'image/jpeg')
        
        print(f"   Tipo de imagem: {mime_type}")
        print(f"   Enviando para GPT-4 Vision...")
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            analysis = response.choices[0].message.content
            print(f"   ✅ Análise recebida ({len(analysis)} caracteres)")
            
            return analysis
            
        except Exception as e:
            print(f"   ❌ Erro ao analisar imagem: {e}")
            raise
