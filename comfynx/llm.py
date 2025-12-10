

class LlmPromptRewrite:
    """
    Dieser ComfyUI-Node `LlmPromptRewrite` dient zur modellgestützten Umformulierung von Texteingaben über die
    OpenAI Chat Completions API. Die Node akzeptiert einen Prompt sowie
    einen Anweisungstext, der den Platzhalter "{prompt}" enthält. Der Platzhalter wird zur
    Laufzeit durch den tatsächlichen Prompt ersetzt und anschließend an das ausgewählte
    OpenAI-Modell gesendet. Die vom Modell gelieferte Neufassung wird als String an den
    ComfyUI-Datenfluss zurückgegeben.

    Inputs
    ------
    prompt : STRING
        Der ursprüngliche, zu überarbeitende Text.

    rewrite_instruction : STRING
        Eine Anweisung zur Umformulierung, die den Platzhalter "{prompt}" enthalten kann.
        Dieser wird vor dem API-Aufruf automatisch ersetzt.

    openai_api_key : STRING
        API-Schlüssel für die Authentifizierung beim OpenAI-Endpunkt.

    model : STRING
        Name des zu verwendenden Chat-Modells (z. B. "gpt-4.1-mini").

    Output
    ------
    STRING
        Die vom Modell generierte, überarbeitete Variante des Prompts.

    Einsatzgebiet
    -------------
    Der Node eignet sich für automatische Textoptimierung, Vereinheitlichung von Schreibstilen,
    Prompt-Engineering-Pipelines sowie alle Workflows, die eine dynamische Umformulierung
    von Benutzereingaben innerhalb von ComfyUI benötigen.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "rewrite_instruction": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"multiline": False}),
                "api_url": ("STRING", {"multiline": False}),
                "api_model": ("STRING", {"multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "rewrite"
    CATEGORY = "LLM"

    def rewrite(self, prompt, rewrite_instruction, api_key, api_url, api_model):
        import openai
        full_instruction = rewrite_instruction.replace("{prompt}", prompt) # Replace placeholder

        client = openai.OpenAI(base_url = api_url, api_key = api_key)

        response = client.chat.completions.create(
            model=api_model,
            messages=[
                {"role": "system", "content": "Rewrite the prompt according to the instructions."},
                {"role": "user", "content": full_instruction}
            ],
            temperature=0.2
        )

        rewritten = response.choices[0].message.content.strip()
        return (rewritten,)


# Node export
NODE_CLASS_MAPPINGS = {
    "LlmPromptRewrite": LlmPromptRewrite
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LlmPromptRewrite": "LLM: Prompt Rewrite"
}
