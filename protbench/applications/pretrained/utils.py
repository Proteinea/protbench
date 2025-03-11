
def shift_embeddings(
    embeddings,
    shift_left: int | None = None,
    shift_right: int | None = None,
):
    if shift_left is not None:
        embeddings = embeddings[:, shift_left:, :]
    if shift_right is not None:
        embeddings = embeddings[:, :-shift_right, :]
    return embeddings
