from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActionGenerator:
    def __init__(self):
        self.base_model_path = "Meta-Llama-3-8B-Instruct"
        self.adaptor_path = "instruction-gen/action-llama3-sft"
        self.device = torch.device("cuda")
        self.model, self.tokenizer = self.load_model_adapter()

    def load_model_adapter(self):
        model_tmp = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            return_dict=True,
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = PeftModel.from_pretrained(model_tmp, self.adaptor_path)
        model = model.merge_and_unload()
        model = model.to(self.device)
        model.eval()
        return model, tokenizer

    def inference_batch(self, messages_list):
        with torch.no_grad():
            all_input_ids = []
            all_attention_masks = []

            for messages in messages_list:
                encoded = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=2048
                )
                all_input_ids.append(encoded)

            #padding
            max_length = max(ids.size(1) for ids in all_input_ids)
            padded_input_ids = []
            attention_masks = []

            for ids in all_input_ids:
                pad_length = max_length - ids.size(1)

                if pad_length > 0:
                    # padding
                    padded_ids = torch.cat([
                        torch.full((1, pad_length), self.tokenizer.pad_token_id, dtype=torch.long),
                        ids
                    ], dim=1)
                    mask = torch.cat([
                        torch.zeros(1, pad_length, dtype=torch.long),
                        torch.ones(1, ids.size(1), dtype=torch.long)
                    ], dim=1)
                else:
                    padded_ids = ids
                    mask = torch.ones_like(ids, dtype=torch.long)

                padded_input_ids.append(padded_ids)
                attention_masks.append(mask)


            batch_input_ids = torch.cat(padded_input_ids, dim=0).to(self.device)
            batch_attention_mask = torch.cat(attention_masks, dim=0).to(self.device)

            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = self.model.generate(
                batch_input_ids,
                attention_mask=batch_attention_mask,
                max_new_tokens=256,
                eos_token_id=terminators,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
            )

            batch_size = batch_input_ids.size(0)
            decoded_responses = []
            for i in range(batch_size):
                generated_ids = outputs[i][batch_input_ids.shape[1]:]
                text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                decoded_responses.append(text)
            torch.cuda.empty_cache()
        return decoded_responses

app = Flask(__name__)
action_generator = ActionGenerator()

@app.route('/infer', methods=['POST'])
def infer():
    try:
        data = request.get_json()
        if not data or "messages_list" not in data:
            return jsonify({"error": "'messages' no data"}), 400

        messages_list = data["messages_list"]
        response = action_generator.inference_batch(messages_list)
        return jsonify({"response": response}), 200

    except Exception as e:
        logger.error(f"request error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    logger.info("Start Flask Service...")
    app.run(host='127.0.0.1', port=6006)