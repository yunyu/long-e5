from typing import Dict, Optional
from sentence_transformers.models import Transformer
from transformers import AutoModel
from transformers.optimization import Adafactor, AdafactorSchedule


class MosaicBertTransformer(Transformer):
    def __init__(
        self,
        model_name_or_path: str = "mosaicml/mosaic-bert-base-seqlen-2048",
        cache_dir: Optional[str] = None,
        model_args: Dict = {},
        tokenizer_args: Dict = {},
        max_seq_length: int = 2048,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            tokenizer_name_or_path="bert-base-uncased",
            model_args={
                **model_args,
                "trust_remote_code": True,
                "auto_map": {"AutoModel": "bert_layers.BertModel"},
            },
            tokenizer_args=tokenizer_args,
            do_lower_case=True,
            max_seq_length=max_seq_length,
            cache_dir=cache_dir,
        )

    def _load_model(self, model_name_or_path, config, cache_dir):
        self.auto_model = AutoModel.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            config=config,
            trust_remote_code=True,
            add_pooling_layer=False,
        )

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        trans_features = {
            "input_ids": features["input_ids"],
            "attention_mask": features["attention_mask"],
        }
        if "token_type_ids" in features:
            trans_features["token_type_ids"] = features["token_type_ids"]

        output_states = self.auto_model(
            **trans_features, output_all_encoded_layers=False
        )
        output_tokens = output_states[0]

        features.update(
            {
                "token_embeddings": output_tokens,
                "attention_mask": features["attention_mask"],
            }
        )
        return features
