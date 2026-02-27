import logging
import os
import base64

from openai import OpenAI
from config import settings

logger = logging.getLogger(__name__)


class ImageAnalyzer:
    """Módulo de Visão Computacional: análise de imagens via GPT-4 Vision."""

    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)

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

    def analyze_image(self, image_path: str) -> str:
        """Analisa uma imagem usando GPT-4 Vision e retorna a análise em texto."""
        logger.info("Analisando imagem: %s", os.path.basename(image_path))

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

        try:
            response = self.client.chat.completions.create(
                model=settings.vision_model,
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
            logger.info("Análise concluída (%d caracteres).", len(analysis))
            return analysis

        except Exception:
            logger.exception("Erro ao analisar imagem.")
            raise
