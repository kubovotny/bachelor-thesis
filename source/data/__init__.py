def partition_indices(df, partition_size=400) -> list[tuple[int, int]]:
    indices = []
    x = 0
    for x in range(partition_size, len(df), partition_size):
        indices.append((x - partition_size, x-1))
    indices.append((x, len(df)))
    return indices