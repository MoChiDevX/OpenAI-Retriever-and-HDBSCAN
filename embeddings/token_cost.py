import tiktoken

# 每个模型的 token 单价（美元 / 千 token）
MODEL_PRICING = {
    "text-embedding-ada-002": 0.0001,
    "text-embedding-babbage-001": 0.0005,
    "text-embedding-curie-001": 0.0020
}

def extract_model_name(full_name):
    return full_name.split("(")[0].strip()

def count_tokens(text_list, model):
    base_model = extract_model_name(model)
    encoding = tiktoken.encoding_for_model(base_model)
    return sum(len(encoding.encode(text)) for text in text_list)

def estimate_cost(total_tokens, model):
    base_model = extract_model_name(model)
    unit_price = MODEL_PRICING.get(base_model, 0.0001)  # fallback to ada
    return (total_tokens / 1000) * unit_price
