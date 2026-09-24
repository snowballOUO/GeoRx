# GeoRx

GeoRx is a source-only package for geometry-aware diagnosis and retrieval repair. It includes the latest 90 encoder-load diagnosis, single repairs, joint repairs, controlled injection checks, reference-model ablation, and joint conformal-threshold ablation. Datasets, checkpoints, generated runs, and caches are excluded.

## Requirements

Run from the GeoRx directory. Replace angle-bracket values with locations chosen by the reader.

```bash
cd <GeoRx-directory>
export GEORX_ROOT="$(pwd)"
export GEORX_DATA_ROOT="<your-hdf5-root>"
export GEORX_MBEIR_ROOT="<optional-raw-data-root>"
export GEORX_PYTHON="${GEORX_PYTHON:-python3}"
export PYTHONPATH="$GEORX_ROOT/main/code:$GEORX_ROOT/injection/code:$GEORX_ROOT/vendor/n1:$GEORX_ROOT/vendor/legacy"
"$GEORX_PYTHON" -m pip install -r requirements.txt
```

Python 3.10 is the reference interpreter. GPU runs require a CUDA-enabled PyTorch build compatible with the installed NVIDIA driver; CPU-only runs need the matching CPU PyTorch wheel.

## Reference model

The sphere reference profiles and sphere gate are built by `stage_diag_recall.py`. Start that stage before native diagnosis:

```bash
"$GEORX_PYTHON" "$GEORX_ROOT/main/code/run_pipeline.py" --stages diag_recall --seeds 1 --device cuda
```

Then run the native 90-pair diagnosis:

```bash
bash "$GEORX_ROOT/main/launch_gate_then_native.sh"
```

The main implementation uses the latest H1-H5 registry: H1 skew and coefficient-of-variation excess, H2 excess pairwise cosine structure, H3 second-neighbor mass and clustering, H4 top-one/top-two gap, and H5 score interaction share.

## HDF5 files

Prepare the embeddings yourself or use an approved embedding-generation pipeline. Set `GEORX_DATA_ROOT` to a directory containing one encoder directory per encoder and one HDF5 file per workload. GeoRx does not download, generate, or store these files.

Each HDF5 file must provide:

- `query/emb`: test query embeddings with shape `[number_of_queries, dimension]`.
- `query/emb_train`: optional training-query embeddings with the same dimension.
- `pool/emb`: candidate embeddings with shape `[number_of_candidates, dimension]`.
- `qrels/query_idx` and `qrels/pool_idx`: parallel integer arrays defining relevant query-candidate pairs.
- An optional `modality` file attribute for cross-encoder routing.

Use float32 or float16 embeddings with a consistent dimension within a file. The mapping in `main/code/cell_splits.yaml` may be edited for a custom HDF5 layout.

## Quick start

Single repairs and the native repair stage:

```bash
bash "$GEORX_ROOT/main/code/launch.sh" --stages repair --seeds 1 2 3 --device cuda
```

H4 listwise training/application:

```bash
"$GEORX_PYTHON" "$GEORX_ROOT/main/code/launch_h4_list.py"
```

Joint repairs use the packaged phase launcher:

```bash
"$GEORX_PYTHON" "$GEORX_ROOT/joint/code/launch_joint_plan.py" --phase A --start
"$GEORX_PYTHON" "$GEORX_ROOT/joint/code/launch_joint_plan.py" --phase B --start
```

The joint code contains the latest `{H1,H2,H4,H5}` and `{H2,H5}` combinations.

Controlled injection smoke test and full run:

```bash
bash "$GEORX_ROOT/injection/code/start_nohup.sh" smoke
bash "$GEORX_ROOT/injection/code/start_nohup.sh" full
```

Reference-model ablation:

```bash
"$GEORX_PYTHON" "$GEORX_ROOT/ablations/reference_model/code/run_reference_model_ablation.py"
```

Joint conformal-threshold ablation:

```bash
"$GEORX_PYTHON" "$GEORX_ROOT/ablations/conformal/code/run_conformal_ablation.py"
```

Set the external checkpoint variables before repair runs:

```bash
export GEORX_LISTWISE_WEIGHTS_ROOT="<your-listwise-weight-root>"
export GEORX_PAIR_WEIGHTS_ROOT="<your-pair-weight-root>"
export GEORX_FROZEN_STRENGTH="<your-frozen-strength-file>"
```

The fixed repair parameters are CSLS k=10 with penalty 0.50; alpha-QE k=10, mixing 0.10, exponent 3; diffusion k=10, coefficient 0.10, one step; and cross-encoder reranking of the top 100 cosine candidates.
