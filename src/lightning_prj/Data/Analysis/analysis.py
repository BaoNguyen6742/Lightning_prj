# %% [markdown]
# # $\text{Import libraries}$

# %% [markdown]
# ## $\text{Installed libraries}$

# %%
# %% [markdown]
# ## $\text{Built-in libraries}$
# %%
from pathlib import Path
from shutil import copyfile, rmtree

import matplotlib.pyplot as plt
import polars as pl
from tqdm.autonotebook import tqdm

# %% [markdown]
# # $\text{Examine data}$

# %%
DATASET_PATH = (
    Path(__file__).resolve().parents[4] / "Dataset" / "Raw" / "popular_street_foods"
)

# %% [markdown]
# ## $\text{Image data stats}$

# %%
raw_stat_data = pl.read_csv(
    DATASET_PATH / "dataset_stats.csv",
    has_header=True,
)

# %%
raw_stat_data.describe()

# %%
plt.pie(
    raw_stat_data["image_count"].to_list(),
    labels=raw_stat_data["class"].to_list(),
)
plt.show()

# %%
raw_stat_data

# %% [markdown]
# resize to (140x140)

# %% [markdown]
# ## $\text{Images}$

# %%
img_list = list(
    str(i.relative_to(DATASET_PATH / "dataset").as_posix())
    for i in (DATASET_PATH / "dataset").rglob("*.[jpeg png]*")
)

# %%
images_df = pl.DataFrame({
    "raw_path": img_list,
})


# %%
images_df = images_df.with_columns(
    pl.col("raw_path").str.split("/").list.to_struct(fields=["class", "file_name"])
).unnest("raw_path")

# %%
images_df.with_row_index()

# %% [markdown]
# ## $\text{Visualize image}$

# %%
fig, axs = plt.subplots(4, 5, figsize=(20, 16), layout="constrained")
axs = axs.flatten()
for i, row in enumerate(images_df.group_by("class").last().to_dicts()):
    img_path = DATASET_PATH / "dataset" / row["class"] / row["file_name"]
    axs[i].imshow(plt.imread(img_path))
    axs[i].set_title(row["class"], fontsize=15)

fig.suptitle("Last Image of Each Class", fontsize=40)
plt.show()

# %% [markdown]
# # $\text{Split data}$

# %%
TRAIN_SIZE = 0.7
VAL_SIZE = 0.2
TEST_SIZE = 0.1


# %%
shuffled_images_df = images_df.sample(fraction=1, shuffle=True, seed=6742).sort("class")

# %%
# %%

# %%
shuffled_images_df = (
    shuffled_images_df.group_by("class")
    .all()
    .with_columns(pl.col("file_name").list.len().alias("image_count"))
    .explode("file_name")
    .with_columns(pl.int_range(1, pl.len().add(1)).over("class").alias("image_index"))
    .sort(pl.all())
    .with_columns(
        pl.when(
            pl.col("image_index") < (TRAIN_SIZE * pl.col("image_count").cast(pl.Int32))
        )
        .then(pl.lit("train"))
        .otherwise(
            pl.when(
                pl.col("image_index")
                < ((TRAIN_SIZE + VAL_SIZE) * pl.col("image_count")).cast(pl.Int32)
            )
            .then(pl.lit("val"))
            .otherwise(pl.lit("test"))
        )
        .alias("split")
    )
)

# %%
Split_path = DATASET_PATH.parents[1] / "Split"
if Split_path.exists():
    print(f"Directory already exists: {Split_path}")
    rmtree(Split_path)
    print(f"Removed existing directory: {Split_path}")
print(f"Creating directory: {Split_path}\n\n")
Split_path.mkdir(parents=True, exist_ok=True)

for split in ["train", "val", "test"]:
    split_path = Split_path / split
    split_path.mkdir(parents=True, exist_ok=True)
    print(f"Creating directory: {split_path}")

# %%
for row in tqdm(
    shuffled_images_df.iter_rows(named=True), total=shuffled_images_df.height
):
    split = row["split"]
    class_name = row["class"]
    file_name = row["file_name"]

    src_path = DATASET_PATH / "dataset" / class_name / file_name
    dest_path = Split_path / split / f"{class_name}_{file_name}"
    copyfile(src_path, dest_path)

# %%
shuffled_images_df.select("class", "file_name", "split").write_csv(
    Split_path / "split_info.csv",
)
