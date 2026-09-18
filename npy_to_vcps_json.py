'''
uses the same code for getting VCPs from a raw path to create a json path for in-game visualization using racingline mod
'''

import json
from pathlib import Path
from trackmania_rl.bng_trajectory import get_vcp_centers_from_file
from config_files import config

vcp_centers = get_vcp_centers_from_file(trajectory_path='ecusa1.npy', config=config)
print(vcp_centers.shape)

output_path = Path("ecusa1_vcp.json")
output_path.write_text(
    json.dumps(vcp_centers.tolist()),
    encoding="utf-8",
)