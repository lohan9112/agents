from litellm import completion

# Cái này sẽ chạy
completion(model="anthropic/claude-3-haiku-20240307",
           messages=[{"role":"user","content":"hi"}])

# Cái này đang bị 404 với bạn
completion(
    model="anthropic/claude-sonnet-4-20250514", 
    messages=[{"role":"user","content":"hi"}]
)