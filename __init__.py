from .batch_condition import CLIPTextEncodeBatch, StringInput, BatchString

NODE_CLASS_MAPPINGS = {
    "CLIP Text Encode (Batch)": CLIPTextEncodeBatch,
    "String Input": StringInput,
    "Batch String": BatchString
}


WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "WEB_DIRECTORY"]