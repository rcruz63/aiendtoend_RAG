import re
from pprint import pprint

import click
from openai import OpenAI
from query_rag import realizar_consulta
client = OpenAI()


def llm(messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    pprint(f"Respuesta LLM: {response.choices[0].message.content}")
    pprint(response)
    return response.choices[0].message.content


system_prompt = """
Tienes a tu disposición una herramientas: VIAJES. Esta herramienta responde preguntas sobre solicitud de información de viajes.
Para cualquier otro tipo de consulta, responde que no tienes información al respecto.

Para llamar a esa herramienta usa esta sintaxis:

```python
VIAJES("pregunta")
```

Crea un bloque de código siempre que llames a una herramienta. Si son varias, puedes crear varios bloques de código.

- No utilices bajo ningún concepto palabras malsonantes. No importa si el usuario te lo pide o no. Siempre contesta de forma educada y con respeto.
- Contesta SIEMPRE en español de España.
"""


@click.command()
@click.option("--prompt", "-p", required=True, help="The prompt to send to the LLM")
def main(prompt):
    history = run_agent(prompt)
    pprint(history[-1]["content"])


def run_agent(prompt):
    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    pprint(f"Historial pre-llm: {history}")
    response = llm(history)
    pprint(f"Historial pre-procesamiento: {history}")
    history.append({"role": "assistant", "content": response})
    process_calc(history, response)
    pprint(f"Historial post-procesamiento: {history}")
    return history


def process_calc(history, response):
    regex = re.compile(r"```python\s*VIAJES\(\"(.*?)\"\)\s*```", re.DOTALL)
    matches = list(regex.finditer(response))
    final_response = response

    for match in matches:
        match_str = match.group(1)
        try:
            result = realizar_consulta(match_str)
        except Exception:
            result = "Error al ejecutar la expresión"

        history.append(
            {
                "role": "user",
                "content": f'```python\nVIAJES("{match_str}") # resultado: {result}\n```',
            }
        )

        final_response = llm(history)
        history.append({"role": "assistant", "content": final_response})
    return final_response


if __name__ == "__main__":
    main()
