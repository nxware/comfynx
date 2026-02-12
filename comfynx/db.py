

import os
import threading

COUNTER_FILE = os.path.join(os.path.dirname(__file__), "counter.txt")
LOCK = threading.Lock()


class PersistentIncID:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_value": (
                    "INT", {
                        "default": 1, "min": 0,
                        "tooltip": f"Persistenter Zähler wird gespeichert in:\n{COUNTER_FILE}"
                    }
                )
            }
        }

    RETURN_TYPES = ("INT", "STRING",)
    RETURN_NAMES = ("id",)
    FUNCTION = "get_next_id"
    CATEGORY = "utils"

    def get_next_id(self, start_value, **kwargs):
        #try:
        #    prompt = kwargs.get("prompt", {})
        #    workflow = prompt.get("workflow", {})
        #    workflow_name = workflow.get("name", workflow_name)
        #except Exception:
        #    pass
        with LOCK:
            if not os.path.exists(COUNTER_FILE):
                current = start_value
            else:
                try:
                    with open(COUNTER_FILE, "r") as f:
                        current = int(f.read().strip())
                except Exception:
                    current = start_value
            next_id = current + 1
            with open(COUNTER_FILE, "w") as f:
                f.write(str(next_id))

        return (next_id,str(next_id))
    

NODE_CLASS_MAPPINGS = {
    "Persistent Increment ID": PersistentIncID
}