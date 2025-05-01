import polars as pl
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
import umap
from embeddings_converter import train_emb_conv
import torch
import numpy as np


model_name1 = "all-MiniLM-L6-v2"
model_name2 = "paraphrase-albert-small-v2"


def data_preparation() -> pl.DataFrame:
    data = fetch_20newsgroups(categories=[
        'soc.religion.christian',
        'talk.politics.guns',
        'talk.politics.mideast',
        'talk.politics.misc',
        'talk.religion.misc'
        ],
        remove=("headers", "footers", "quotes")
    )
    model = SentenceTransformer(model_name1)
    embeddings = model.encode(data.data)
    model2 = SentenceTransformer(model_name2)
    embeddings_model2 = model2.encode(data.data)

    emb_conv_model = train_emb_conv(input_embeddings=embeddings_model2, output_embeddings=embeddings)
    embeddings_of_model2_as_model1 = emb_conv_model(torch.Tensor(embeddings_model2))
    input_for_umap = np.concat([embeddings, embeddings_of_model2_as_model1.detach().numpy()])
    reducer = umap.UMAP()
    embeddings_2d = reducer.fit_transform(input_for_umap)

    return pl.DataFrame({
        'text': data.data,
        'target': map(lambda i: data.target_names[i], data.target),
        'embeddings_2d': embeddings_2d[:len(embeddings)],
        'embeddings_2d_model2': embeddings_2d[len(embeddings):],
    })


if __name__ == "__main__":
    df = data_preparation()
    print(df.shape)
    print(df.head())
