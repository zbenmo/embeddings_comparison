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
    from data_preparation import data_preparation
    import altair as alt
    return alt, data_preparation, mo, pl


@app.cell
def _(data_preparation):
    df = data_preparation()
    print(df.shape)
    print(df.head())
    return (df,)


@app.cell
def _(df, pl):
    df1 = df.with_columns(
        x=pl.col('embeddings_2d').arr.get(0),
        y=pl.col('embeddings_2d').arr.get(1),
        x_model2=pl.col('embeddings_2d_model2').arr.get(0),
        y_model2=pl.col('embeddings_2d_model2').arr.get(1),
    )
    df1 = df1.filter(pl.col('text').str.len_chars() > 7) # remove empty texts
    return (df1,)


@app.cell
def _(alt, df1):
    chart1 = alt.Chart(df1).mark_point(size=4).encode(
        alt.X("x", axis=alt.Axis(labels=False, ticks=False)),
        alt.Y("y", axis=alt.Axis(labels=False, ticks=False)),
        color='target', # alt.value("red"), #
        tooltip='text',
    )

    chart2 = alt.Chart(df1).mark_point(size=4).encode(
        alt.X("x_model2", axis=alt.Axis(labels=False, ticks=False)),
        alt.Y("y_model2", axis=alt.Axis(labels=False, ticks=False)),
        color='target', #alt.value("blue"), # 
        tooltip='text',
    )
    return chart1, chart2


@app.cell
def _(alt, df1, pl):
    # Connecting Lines
    df_lines = pl.DataFrame({
        "x": df1["x"].to_list() + df1["x_model2"].to_list(),
        "y": df1["y"].to_list() + df1["y_model2"].to_list(),
        "group": list(range(len(df1))) * 2
    })

    lines = alt.Chart(df_lines).mark_line().encode(
        alt.X("x"),
        alt.Y("y"),
        detail="group",
        color=alt.value("gray")
    )
    return (lines,)


@app.cell
def _(chart1, chart2, lines, mo):
    chart = mo.ui.altair_chart(chart1 + chart2 + lines)


    # .properties(
    #     width=600,  # Set width
    #     height=800  # Set height
    # )
    return (chart,)


@app.cell
def _(chart):
    chart
    return


@app.cell
def _(chart, mo):
    table = mo.ui.table(chart.value[['text', 'target']])
    return (table,)


@app.cell
def _(table):
    table
    return


@app.cell
def _(table):
    table.value
    return


@app.cell
def _(df1, pl):
    distance_exp = (
        (pl.col('embeddings_2d') - pl.col('embeddings_2d_model2')
    ).cast(pl.List(pl.Float32)).list.eval(pl.element() ** 2)).list.sum().sqrt()

    df2 = df1.with_columns(distance=distance_exp)
    return (df2,)


@app.cell
def _(df2, pl):
    df2.sort(pl.col('distance'), descending=True)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
