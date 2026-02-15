import torch
import torch.nn.functional as F
from tokenizer import AutoTokenizer
import matplotlib.pyplot as plt
import io
# import base64
# from PIL import Image
import numpy as np
# NOTE THIS WORKS FOR PLOTTING LANDSCAPE JUST DOESNT DO PER LANDSCAPE FOR TIME EMBED MODELS
# most of this code is from https://github.com/meta-llama/llama/blob/main/llama/generation.py#L129

def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def call_model_forward_decode(hparams, model, input_tokens, start_pos, bsz):
    #TODO eventually add back kv caching, for now start_pos is not supported  in baseline transformer and EBT so start_pos can only be 0
    if hparams.model_name == "ebt":
        if hparams.infer_ebt_advanced:
            ebt_outputs = model.ebt_advanced_inference(input_tokens, start_pos = 0, learning = False)
            logits = ebt_outputs[0] # dont return a list just return the final predicted logits
        else:
            ebt_outputs = model.forward(input_tokens, start_pos = 0, learning = False, return_raw_logits = True)
            logits = ebt_outputs[0][-1] # uses 0, -1 since ebt returns tuple of lists of (logits, energy predictions) for each mcmc step; dont want learning mode since needs grad
        energies = ebt_outputs[1]
        energies = [energy_tensor.reshape(bsz, -1).mean(dim=1) for energy_tensor in energies] # will be num_mcmc_step * energy landscapes len list, with bsz elements each
    else:
        logits = model.forward(input_tokens, start_pos = 0, learning = False, return_raw_logits = True)
    return logits

def call_model_forward_ppl(hparams, model, input_tokens, start_pos, bsz):
    #TODO same issues with start pos as above
    if hparams.model_name == "ebt":
        if hparams.infer_ebt_advanced:
            ebt_outputs = model.ebt_advanced_inference(input_tokens, start_pos = 0, learning = False)
            logits = ebt_outputs[0] # dont return a list just return the final predicted logits
        else:
            ebt_outputs = model.forward(input_tokens, start_pos = 0, learning = False, return_raw_logits = True)
            logits = ebt_outputs[0][-1] # uses 0, -1 since ebt returns tuple of lists of (logits, energy predictions) for each mcmc step; dont want learning mode since needs grad
        energies = ebt_outputs[1]
        energies = [energy_tensor.reshape(bsz, -1).mean(dim=1) for energy_tensor in energies] # will be num_mcmc_step * energy landscapes len list, with bsz elements each
    else:
        logits = model.forward(input_tokens, start_pos = 0, learning = False, return_raw_logits = True)
        energies = None
    return logits, energies

def generate_text(model, batch, hparams):
    tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer, clean_up_tokenization_spaces = False)
    tokenizer_pad_token_id = tokenizer.eos_token_id # is token 0, was right padding things
        
    questions, answers = batch
    ids = questions['input_ids']
    attn_mask = questions["attention_mask"]
    max_gen_len = hparams.infer_max_gen_len
    temperature = hparams.infer_temp
    top_p = hparams.infer_topp
    logprobs = hparams.infer_logprobs
    echo = hparams.infer_echo
    # ppl = model.forward_loss_wrapper(questions, phase="test")['perplexity'].item() # just in case want to debug model PPL

    prompt_tokens = [] #NOTE this was to fix a bug where this generation code was not working for bs > 1 due to pad_token_id being same as eos_token_id and min_prompt_len being wrong
    for row_ids, row_mask in zip(ids, attn_mask):
        seq_len = row_mask.sum().item()         # number of *real* tokens
        prompt_tokens.append(row_ids[:seq_len].tolist())
    
    params = model.transformer.params
    bsz = len(prompt_tokens)
    assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

    min_prompt_len = min(len(t) for t in prompt_tokens)
    max_prompt_len = max(len(t) for t in prompt_tokens)
    # if max_prompt_len > hparams.context_length:
    #     over_length_prompt = max(prompt_tokens, key=len)
    #     print(f"Prompt exceeding max length ({max_prompt_len} > {hparams.context_length}):")
    #     print(tokenizer.decode(over_length_prompt))
    assert max_prompt_len <= hparams.context_length
    total_len = min(hparams.context_length, max_gen_len + max_prompt_len)
    pad_id = tokenizer_pad_token_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda").clone().detach()
    if logprobs:
        token_logprobs = torch.zeros_like(tokens, dtype=torch.float)
    prev_pos = 0
    eos_reached = torch.tensor([False] * bsz, device="cuda")
    input_text_mask = tokens != pad_id
    with torch.no_grad():
        if min_prompt_len == total_len:
            logits = call_model_forward_decode(hparams, model, tokens, prev_pos, bsz)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )
        for cur_pos in range(min_prompt_len, total_len):
            input_tokens = tokens[:, :cur_pos] # NOTE removed prev_pos since are not using start_pos in model forward for now, TODO eventually add back
            logits = call_model_forward_decode(hparams, model, input_tokens, prev_pos, bsz)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)
            
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == tokenizer.eos_token_id
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

    if logprobs:
        token_logprobs = token_logprobs.tolist()
    out_tokens, out_logprobs = [], []
    for i, toks in enumerate(tokens.tolist()):
        # cut to max gen len
        start = 0 if echo else len(prompt_tokens[i])
        toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
        probs = None
        if logprobs:
            probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
        # cut to eos tok if any
        if tokenizer.eos_token_id in toks:
            eos_idx = toks.index(tokenizer.eos_token_id)
            toks = toks[:eos_idx]
            probs = probs[:eos_idx] if logprobs else None
        out_tokens.append(toks)
        out_logprobs.append(probs)

    if logprobs:
        return [
            {
                "generation": tokenizer.decode(t, skip_special_tokens=True),
                "tokens": [tokenizer.decode(x) for x in t],
                "logprobs": logprobs_i,
                "gt_answer": tokenizer.decode(gt_ans, skip_special_tokens=True),
                "question": tokenizer.decode(question, skip_special_tokens=True),
            }
            for t, logprobs_i, gt_ans, question in zip(out_tokens, out_logprobs, answers['input_ids'], questions['input_ids'])
        ]
    return [
        {
            "generation": tokenizer.decode(t, skip_special_tokens=True),
            "gt_answer": tokenizer.decode(gt_ans, skip_special_tokens=True),
            "question": tokenizer.decode(question, skip_special_tokens=True),
        } for t, gt_ans, question in zip(out_tokens, answers['input_ids'], questions['input_ids'])
    ]


def get_ppl(model, batch, hparams): # is very similar to model forward_loss_wrapper, just doesnt work on list for logits and calls advanced inference. mainly to avoid using inference_mode which generates tokens one at a time, this is for getting PPL over the entire sequence
    batch_size = batch['input_ids'].shape[0]
    with torch.no_grad(): # by default no grad, although ebt will enable grad
        input_ids = batch['input_ids'].squeeze(dim=1)[:, :-1]
        if hparams.model_name == "ebt":
            logits, energies = call_model_forward_ppl(hparams, model, input_ids, 0, batch_size)
        else:
            logits, _ = call_model_forward_ppl(hparams, model, input_ids, 0, batch_size) 

    next_token_indices = batch['input_ids'].squeeze(dim=1)[:, 1:].reshape(-1) # BS * S; reshape since targets are supposed to be 1D
    cce_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), next_token_indices, ignore_index=model.tokenizer_pad_token_id)
    perplexity = torch.exp(cce_loss).detach()

    outputs = {
        'loss': cce_loss,
        'perplexity': perplexity
    }

    if hparams.model_name == "ebt":
        energy_tensors = []
        for step_energies in zip(*[energies]):
            step_tensor = torch.stack(step_energies)
            avg_step_energy = torch.mean(step_tensor, dim=0)
            energy_tensors.append(avg_step_energy)
        
        for step_idx, energy in enumerate(energy_tensors):
            outputs[f"mcmc_step_{step_idx}_energy"] = energy

    if hparams.infer_plot_energy_landscape:
        assert hparams.model_name == "ebt", "Energy landscape plotting only works with EBT models"
        
        _, energies_list, predicted_tokens_list = model.ebt_advanced_inference(input_ids, start_pos=0, learning=False)
        
        # Create separate plots for different MCMC steps
        image_tensors = {}
        
        # Select steps to plot (first, middle, and last step)
        num_steps = len(predicted_tokens_list)
        steps_to_plot = list(range(num_steps))
        
        for step_idx in steps_to_plot:
            token_losses = []
            step_energies = []
            
            # Get target tokens (actual next tokens in sequence)
            target_tokens = batch['input_ids'].squeeze(dim=1)[:, 1:]
            
            for batch_idx in range(batch_size):
                for pos_idx in range(input_ids.shape[1]):
                    # Skip padded positions
                    if pos_idx >= target_tokens.shape[1] or target_tokens[batch_idx, pos_idx] == model.tokenizer_pad_token_id:
                        continue
                    
                    target_token = target_tokens[batch_idx, pos_idx]
                    
                    # Get predicted token distribution at this position for this step
                    token_logit = predicted_tokens_list[step_idx][batch_idx, pos_idx]
                    
                    # Calculate cross entropy loss for this prediction vs ground truth
                    token_loss = F.cross_entropy(
                        token_logit.unsqueeze(0), 
                        target_token.unsqueeze(0),
                        reduction='none'
                    ).item()
                    
                    # Get energy value for this position at this step
                    token_energy = energies_list[step_idx][batch_idx, pos_idx].item()
                    
                    token_losses.append(token_loss)
                    step_energies.append(token_energy)
            
            # Create the scatter plot for this step
            plt.figure(figsize=(10, 6))
            plt.scatter(step_energies, token_losses, alpha=0.5)
            plt.xlabel('Predicted Energy')
            plt.ylabel('Ground Truth Cross-Entropy Loss')
            plt.title(f'Energy Landscape vs Ground Truth Loss (MCMC Step {step_idx})')
            
            # Add trend line if there are enough points
            if len(step_energies) > 5:
                z = np.polyfit(step_energies, token_losses, 1)
                p = np.poly1d(z)
                plt.plot(sorted(step_energies), p(sorted(step_energies)), "r--", alpha=0.8)
                
                # Add correlation coefficient
                from scipy.stats import pearsonr
                corr, _ = pearsonr(step_energies, token_losses)
                plt.annotate(f"Correlation: {corr:.3f}", xy=(0.05, 0.95), xycoords='axes fraction')
            
            # Save plot to buffer and convert to tensor
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            
            # # Convert to PIL Image and then to tensor
            # pil_img = Image.open(buf).convert('RGB')
            # # Convert PIL image to tensor (channels, height, width) with values in [0, 1]
            # img_tensor = torch.FloatTensor(np.array(pil_img)).permute(2, 0, 1) / 255.0
            
            # image_tensors[f"image_energy_landscape_step_{step_idx}"] = img_tensor
        
        # Also create a combined plot showing the evolution
        plt.figure(figsize=(10, 6))
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']
        
        for i, step_idx in enumerate(steps_to_plot):
            token_losses = []
            step_energies = []
            
            for batch_idx in range(batch_size):
                for pos_idx in range(input_ids.shape[1]):
                    if pos_idx >= target_tokens.shape[1] or target_tokens[batch_idx, pos_idx] == model.tokenizer_pad_token_id:
                        continue
                    
                    target_token = target_tokens[batch_idx, pos_idx]
                    token_logit = predicted_tokens_list[step_idx][batch_idx, pos_idx]
                    
                    token_loss = F.cross_entropy(
                        token_logit.unsqueeze(0), 
                        target_token.unsqueeze(0),
                        reduction='none'
                    ).item()
                    
                    token_energy = energies_list[step_idx][batch_idx, pos_idx].item()
                    
                    token_losses.append(token_loss)
                    step_energies.append(token_energy)
            
            color_idx = i % len(colors)
            plt.scatter(step_energies, token_losses, alpha=0.5, color=colors[color_idx], 
                        label=f'Step {step_idx}')
        
        plt.xlabel('Predicted Energy')
        plt.ylabel('Ground Truth Cross-Entropy Loss')
        plt.title('Energy Landscape Evolution During MCMC Steps')
        plt.legend()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        # pil_img = Image.open(buf).convert('RGB')
        # img_tensor = torch.FloatTensor(np.array(pil_img)).permute(2, 0, 1) / 255.0
        
        # image_tensors["image_energy_landscape_combined"] = img_tensor
        
        # # Add all images to outputs
        # for key, tensor in image_tensors.items():
        #     outputs[key] = tensor

    return outputs