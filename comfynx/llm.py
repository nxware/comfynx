

class LlmPromptRewrite:
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
