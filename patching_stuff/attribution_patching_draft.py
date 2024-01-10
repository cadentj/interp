# %%

import nnsight 
import torch
import plotly.express as px
import einops

# %%

model = nnsight.LanguageModel("gpt2", device_map="cuda:0")
tokenizer = model.tokenizer
# %%

prompts = ['When John and Mary went to the shops, John gave the bag to', 'When John and Mary went to the shops, Mary gave the bag to', 'When Tom and James went to the park, James gave the ball to', 'When Tom and James went to the park, Tom gave the ball to', 'When Dan and Sid went to the shops, Sid gave an apple to', 'When Dan and Sid went to the shops, Dan gave an apple to', 'After Martin and Amy went to the park, Amy gave a drink to', 'After Martin and Amy went to the park, Martin gave a drink to']
answers = [(' Mary', ' John'), (' John', ' Mary'), (' Tom', ' James'), (' James', ' Tom'), (' Dan', ' Sid'), (' Sid', ' Dan'), (' Martin', ' Amy'), (' Amy', ' Martin')]

clean_tokens = [tokenizer.encode(prompt) for prompt in prompts]
clean_tokens = [torch.tensor(clean_token_str, device="cuda:0") for clean_token_str in clean_tokens]
clean_tokens = torch.stack(clean_tokens)

# %%

# Swap each adjacent pair, with a hacky list comprehension
corrupted_tokens = clean_tokens[
    [(i+1 if i%2==0 else i-1) for i in range(len(clean_tokens)) ]
    ]
print("Clean string 0", tokenizer.decode(clean_tokens[0]))
print("Corrupted string 0", tokenizer.decode(corrupted_tokens[0]))

# %%

answer_token_indices = torch.tensor([[tokenizer.encode(answers[i][j])[0] for j in range(2)] for i in range(len(answers))], device="cuda:0")
print("Answer token indices", answer_token_indices)

# %%

def get_logit_diff(logits, answer_token_indices=answer_token_indices):
    if len(logits.shape)==3:
        # Get final logits only
        logits = logits[:, -1, :]
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    return (correct_logits - incorrect_logits).mean()

# %%

with model.invoke(clean_tokens) as clean_invoker:
    pass

with model.invoke(corrupted_tokens) as corrupted_invoker:
    pass

# %%

clean_logits = clean_invoker.output.logits
corrupted_logits = corrupted_invoker.output.logits

clean_logit_diff = get_logit_diff(clean_logits, answer_token_indices).item()
print(f"Clean logit diff: {clean_logit_diff:.4f}")

corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_token_indices).item()
print(f"Corrupted logit diff: {corrupted_logit_diff:.4f}")


# %%

CLEAN_BASELINE = clean_logit_diff
CORRUPTED_BASELINE = corrupted_logit_diff
def ioi_metric(logits, answer_token_indices=answer_token_indices):
    return (get_logit_diff(logits, answer_token_indices) - CORRUPTED_BASELINE) / (CLEAN_BASELINE  - CORRUPTED_BASELINE)

print(f"Clean Baseline is 1: {ioi_metric(clean_logits).item():.4f}")
print(f"Corrupted Baseline is 0: {ioi_metric(corrupted_logits).item():.4f}")



# %%

def get_forward_and_backward_caches(model, clean, corrupted, metric): 

    clean_resid_out = []
    with model.invoke(clean, fwd_args = {"inference": False}) as invoker_clean:
        for layer in model.transformer.h:
            l = layer.output[0].save()
            l.retain_grad()
            clean_resid_out.append(l)
    
    corrupted_resid_out = []

    with model.invoke(corrupted, fwd_args = {"inference": False}) as invoker_corrupted:
        for layer in model.transformer.h:
            l = layer.output[0].save()
            l.retain_grad()
            corrupted_resid_out.append(l)
    
    logits = invoker_corrupted.output.logits
    value = metric(logits)
    value.backward()

    return value.item(), clean_resid_out, corrupted_resid_out

# %%

value, clean_resid_out, corrupted_resid_out = get_forward_and_backward_caches(model, clean_tokens, corrupted_tokens, ioi_metric)


# %%

patching_results = []
for clean, corrupted in zip(clean_resid_out, corrupted_resid_out):
    residual_attr = einops.reduce(
        corrupted.value.grad * (clean.value - corrupted.value),
        "batch pos d_model -> pos",
        "sum"
    )
    patching_results.append(residual_attr.detach().cpu().numpy())



# %%
px.imshow(patching_results, color_continuous_scale="RdBu", color_continuous_midpoint=0.0, title="Patching Results")

