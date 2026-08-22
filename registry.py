from models import CNN, MLP, CNNRevised, MLPRevised, TransformerEncoderModel, TransformerEncoderRevised

MODEL_REGISTRY = {
    "mlp": MLP,
    "cnn": CNN,
    "transformer": TransformerEncoderModel,
    "mlp_revised": MLPRevised,
    "cnn_revised": CNNRevised,
    "transformer_revised": TransformerEncoderRevised,
}

# (checkpoint filename, architecture key) for every frozen model carried over from the
# classification project — checkpoint names add an "_aug" suffix for the augmented-training
# runs, which doesn't correspond to a separate architecture/registry key.
CHECKPOINTS = [
    ("mlp", "mlp"),
    ("mlp_aug", "mlp"),
    ("mlp_revised", "mlp_revised"),
    ("mlp_revised_aug", "mlp_revised"),
    ("cnn", "cnn"),
    ("cnn_aug", "cnn"),
    ("cnn_revised", "cnn_revised"),
    ("cnn_revised_aug", "cnn_revised"),
    ("transformer", "transformer"),
    ("transformer_aug", "transformer"),
    ("transformer_revised", "transformer_revised"),
    ("transformer_revised_aug", "transformer_revised"),
]
