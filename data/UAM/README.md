# Underwater Anomaly Monitoring (UAM)

UAM contains **5,373 RGB images** with pixel-level annotations:
**4,298 images for training and validation** and **1,075 test images**.
Example images and annotations are shown in [Fig. 3 in the repository README](../../README.md#uam-dataset).

~~~text
UAM/
  train/images/<name>.jpg
  train/masks/<name>.png
  test/images/<name>.jpg
  test/masks/<name>.png
  splits/train.json
  splits/test.json
  manifest.jsonl
~~~

| Index | Class | Examples |
|---|---|---|
| 0 | nature | Waterbody, animals, plants, seafloor, wreck |
| 1 | human | Diver, swimmer |
| 2 | vehicle | AUV, ROV, ship, submarine |

Masks are single-channel uint8 class-index PNGs with values 0, 1 and 2.
The paper visualizes these classes as black, red and yellow, respectively.
Image dimensions are preserved. `manifest.jsonl` records checksums and dimensions;
`splits/*.json` lists the samples in each split.

The paper describes image collection from SUIM, WebUOT-1M, COU and public image
platforms, including Wikimedia Commons. Author-created annotations are licensed
under CC BY 4.0; original images retain their source terms.
See [dataset terms](../../DATA_LICENSE.md).
