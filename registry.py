from models import CNN, MLP, CNNRevised, MLPRevised, TransformerEncoderModel, TransformerEncoderRevised

MODEL_REGISTRY = {
    "mlp": MLP,
    "cnn": CNN,
    "transformer": TransformerEncoderModel,
    "mlp_revised": MLPRevised,
    "cnn_revised": CNNRevised,
    "transformer_revised": TransformerEncoderRevised,
}
