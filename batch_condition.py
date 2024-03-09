import torch
import math

def lcm(a, b):
    return a * b // math.gcd(a, b)

def lcm_for_list(numbers):
    current_lcm = numbers[0]
    for number in numbers[1:]:
        current_lcm = lcm(current_lcm, number)
    return current_lcm

class CLIPTextEncodeBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ), 
                "texts":("BATCH_STRING", )
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "conditioning_batch"

    def encode(self, clip, texts):
        conds = []
        pooleds = []
        num_tokens = []
        for text in texts:
            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            conds.append(cond)
            pooleds.append(pooled)
            num_tokens.append(cond.shape[1])
        
        # Make number of tokens equal
        # attn(q, k, v) == attn(q, [k]*n, [v]*n)
        lcm = lcm_for_list(num_tokens)
        repeats = [lcm//num for num in num_tokens]
        conds = torch.cat([cond.repeat(1, repeat, 1) for cond, repeat in zip(conds, repeats)])
        pooleds = torch.cat(pooleds)
        return ([[conds, {"pooled_output": pooleds}]], )
    
class StringInput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
            {
                "text": ("STRING", {"multiline": True})
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "encode"

    CATEGORY = "conditioning_batch"

    def encode(self, text):
        return (text, )
    
class BatchString:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}
    RETURN_TYPES = ("BATCH_STRING",)
    FUNCTION = "encode"

    CATEGORY = "conditioning_batch"

    def encode(self, **kwargs):
        return ([kwargs[f"text{i+1}"] for i in range(len(kwargs))], )
    