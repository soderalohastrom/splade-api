from flask import Flask, request, jsonify
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

app = Flask(__name__)

# Initialize the SPLADE model
sparse_model = AutoModelForMaskedLM.from_pretrained("naver/splade-cocondenser-ensembledistil")

def convert_sparse_values(sparse_values):
    converted_values = []
    for idx, weight in sparse_values.items():
        converted_values.append({
            "idx": idx,
            "weight": weight
        })
    return converted_values

@app.route('/sparse-vectors', methods=['GET'])
def sparse_vectors():
    # Get the 'phrase' parameter from the URL
    phrase = request.args.get('phrase')

    # If no phrase was provided, return an error message
    if phrase is None:
        return "Error: No phrase provided. Please specify a phrase.", 400

    # Generate sparse values using SPLADE
    tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
    tokens = tokenizer(phrase, return_tensors='pt')
    output = sparse_model(**tokens)
    vec = torch.max(torch.log(1 + torch.relu(output.logits)) * tokens.attention_mask.unsqueeze(-1), dim=1)[0].squeeze()
    cols = vec.nonzero().squeeze().cpu().tolist()
    weights = vec[cols].cpu().tolist()
    sparse_values = dict(zip(cols, weights))
    sparse_values_formatted = convert_sparse_values(sparse_values)

    # map token IDs to human-readable tokens
    idx2token = {idx: token for token, idx in tokenizer.get_vocab().items()}
    sparse_dict_tokens = {idx2token[idx]: round(weight, 2) for idx, weight in zip(cols, weights)}

    # sort so we can see most relevant tokens first
    sparse_dict_tokens = {k: v for k, v in sorted(sparse_dict_tokens.items(), key=lambda item: item[1], reverse=True)}

    return jsonify({
        "sparse_values": sparse_values_formatted,
        "tokens": sparse_dict_tokens
    })

if __name__ == '__main__':
    app.run()
