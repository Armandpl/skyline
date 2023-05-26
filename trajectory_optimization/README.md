# Note about designing tracks in Fusion360 to export as dxf

Polyline are easier to parse from dxf files because they don't require matching the start and end of lines and arcs. However, there is no polyline tool in Fusion360, the trick is to create a polygon and delete it's center constraint, then you can freely move the polygon points.

# Ressources

- bicycle model code: https://github.com/r-pad/aa_simulation/
- vehicle model openpilot: https://github.com/commaai/openpilot/blob/16f0f1561b716cbb75149185dac31b15386e7984/selfdrive/controls/lib/vehicle_model.py
- robust sampling for init position: https://codereview.stackexchange.com/questions/69833/generate-sample-coordinates-inside-a-polygon
- RL https://arxiv.org/abs/2008.07971
