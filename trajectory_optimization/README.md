## How to train

Configure you experiment under `scripts/configs/experiments` then run `python train +experiment=your_experiment`

Sometime the whole experiment config is in the env params and you can go `python train env=longitudinal_slow`

Run `python train --cfg job` to print and debug your config.

# Note about designing tracks in Fusion360 to export as dxf

Polyline are easier to parse from dxf files because they don't require matching the start and end of lines and arcs. However, there is no polyline tool in Fusion360, the trick is to create a polygon and delete it's center constraint, then you can freely move the polygon points.
Second trick to make polylines in f360 is to extrude the sketch and then project the body onto another sketch.

# Ressources

- bicycle model code: https://github.com/r-pad/aa_simulation/
- vehicle model openpilot: https://github.com/commaai/openpilot/blob/16f0f1561b716cbb75149185dac31b15386e7984/selfdrive/controls/lib/vehicle_model.py
- robust sampling for init position: https://codereview.stackexchange.com/questions/69833/generate-sample-coordinates-inside-a-polygon
- [Super-Human Performance in Gran Turismo Sport Using Deep Reinforcement Learning](https://arxiv.org/abs/2008.07971) got the reward from here
