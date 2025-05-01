# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from sklearn.datasets import fetch_20newsgroups
    from sentence_transformers import SentenceTransformer
    import umap
    import altair as alt
    from embeddings_converter import (
        EmbeddingConverter, train_emb_conv
    )
    import torch
    import numpy as np
    return (
        SentenceTransformer,
        alt,
        fetch_20newsgroups,
        mo,
        pl,
        torch,
        np,
        EmbeddingConverter,
        train_emb_conv,
        umap,
    )


@app.cell
def _(fetch_20newsgroups):
    # ['alt.atheism',
    #  'comp.graphics',
    #  'comp.os.ms-windows.misc',
    #  'comp.sys.ibm.pc.hardware',
    #  'comp.sys.mac.hardware',
    #  'comp.windows.x',
    #  'misc.forsale',
    #  'rec.autos',
    #  'rec.motorcycles',
    #  'rec.sport.baseball',
    #  'rec.sport.hockey',
    #  'sci.crypt',
    #  'sci.electronics',
    #  'sci.med',
    #  'sci.space',
    #  'soc.religion.christian',
    #  'talk.politics.guns',
    #  'talk.politics.mideast',
    #  'talk.politics.misc',
    #  'talk.religion.misc']

    data = fetch_20newsgroups(categories=[
     'soc.religion.christian',
     'talk.politics.guns',
     'talk.politics.mideast',
     'talk.politics.misc',
     'talk.religion.misc'
    ],
    remove=("headers", "footers", "quotes")
    )
    return (data,)


@app.cell
def _(data):
    data.keys()
    return


@app.cell
def _():
    model_name1 = "all-MiniLM-L6-v2"
    model_name2 = "paraphrase-albert-small-v2"
    return model_name1, model_name2


@app.cell
def _(SentenceTransformer, model_name1):
    model = SentenceTransformer(model_name1)
    return (model,)


@app.cell
def _(data, model):
    embeddings = model.encode(data.data)
    return (embeddings,)


@app.cell
def _(embeddings):
    embeddings.shape
    return


@app.cell
def _(data, pl, umap):
    def prepare_df(embeddings):
        reducer = umap.UMAP()
        embeddings_2d = reducer.fit_transform(embeddings)
        return pl.DataFrame({
            'dim_0': embeddings_2d[:, 0],
            'dim_1': embeddings_2d[:, 1],
            'text': data.data,
            'target': map(lambda i: data.target_names[i], data.target),
        })
    return (prepare_df,)


@app.cell
def _(embeddings, prepare_df):
    df = prepare_df(embeddings)
    return (df,)


@app.cell
def _(alt, df, mo):
    chart = mo.ui.altair_chart(alt.Chart(df).mark_point(size=4).encode(
        alt.X('dim_0', axis=alt.Axis(labels=False, ticks=False)),
        alt.Y('dim_1', axis=alt.Axis(labels=False, ticks=False)),
        color='target',
        tooltip='text',
    ))
    return (chart,)


@app.cell
def _(chart, mo):
    table = mo.ui.table(chart.value[['text', 'target']])
    return (table,)


@app.cell
def _(chart, mo, table):
    mo.vstack([
        chart, table
    ])
    return


@app.cell
def _(table):
    table.value
    return


@app.cell
def _(SentenceTransformer, model_name2):
    model2 = SentenceTransformer(model_name2)
    return (model2,)


@app.cell
def _(data, model2):
    embeddings_model2 = model2.encode(data.data)
    return (embeddings_model2,)


@app.cell
def _(embeddings_model2):
    embeddings_model2.shape
    return


@app.cell
def _(embeddings_model2, prepare_df):
    df_model2 = prepare_df(embeddings_model2)
    return (df_model2,)


@app.cell
def _(alt, df_model2, mo):
    chart_model2 = mo.ui.altair_chart(alt.Chart(df_model2).mark_point(size=4).encode(
        alt.X('dim_0', axis=alt.Axis(labels=False, ticks=False)),
        alt.Y('dim_1', axis=alt.Axis(labels=False, ticks=False)),
        color='target',
        tooltip='text',
    ))
    return (chart_model2,)


@app.cell
def _(chart_model2, mo):
    table_model2 = mo.ui.table(chart_model2.value[['text', 'target']])
    return (table_model2,)


@app.cell
def _(chart_model2, mo, table_model2):
    mo.vstack([
        chart_model2, table_model2
    ])
    return


@app.cell
def _(table_model2):
    table_model2.value
    return


@app.cell
def _(embeddings, embeddings_model2):
    print(embeddings.shape)
    print(embeddings_model2.shape)
    return


@app.cell
def _(embeddings, embeddings_model2, train_emb_conv):
    emb_conv_model = train_emb_conv(embeddings, embeddings_model2)
    return (emb_conv_model,)


@app.cell
def _(emb_conv_model, embeddings, torch):
    embeddings_as_model2 = emb_conv_model(torch.Tensor(embeddings))
    return (embeddings_as_model2,)


@app.cell
def _(embeddings_as_model2, embeddings_model2, prepare_df, np):

    df_together = prepare_df(np.concat([embeddings_model2, embeddings_as_model2.detach().numpy()]))
    return (df_together,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
