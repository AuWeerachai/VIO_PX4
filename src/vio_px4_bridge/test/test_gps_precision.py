"""Check configured confidence and the actual HIL_GPS wire scaling without ROS."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from vio_px4_bridge.geo_utils import course_over_ground_cdeg


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'src/vio_px4_bridge/vio_px4_bridge/vio_px4_gps_bridge.py'


def test_bridge_and_launcher_use_requested_normal_confidence():
    tree = ast.parse(SOURCE.read_text())
    defaults = {
        n.args[0].value: ast.literal_eval(n.args[1])
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == 'declare_parameter' and len(n.args) == 2
    }
    launcher = ast.parse((ROOT / 'scripts/vio-launch/vio_launch.py').read_text())
    for key in ('boot_accuracy_m', 'cruise_eph_m'):
        assert defaults[key] == 0.0111
        values = [ast.literal_eval(value) for node in ast.walk(launcher)
                  if isinstance(node, ast.Dict)
                  for name, value in zip(node.keys, node.values)
                  if isinstance(name, ast.Constant) and name.value == key]
        assert values == [0.0111]


def test_wire_preserves_standard_coordinate_scaling_and_one_cm_eph():
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'VioPx4GpsBridge')
    method = next(n for n in cls.body if n.name == '_publish_hil_gps')
    clamp = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == 'clamp_int')
    ns = dict(math=math, time=SimpleNamespace(time=lambda: 123.),
              course_over_ground_cdeg=course_over_ground_cdeg)
    exec(compile(ast.Module(body=[clamp, method], type_ignores=[]), str(SOURCE), 'exec'), ns)
    bridge = SimpleNamespace(fix_type=3, satellites=40, gps_id=1,
                             _hil_gps_extensions_supported=None, _mav=Mock())
    fix = dict(lat=40.41358964, lon=-79.94629414, alt=200., eph=.0111,
               epv=1., vn=0., ve=0., vd=0.)
    ns['_publish_hil_gps'](bridge, fix)
    args = bridge._mav.mav.hil_gps_send.call_args.args
    assert args[2:4] == (404135896, -799462941)
    assert args[5] == 1  # PX4 decodes this as 0.01 m, not zero.
    fix['eph'] = 0.37  # Recovery uncertainty must still be transmitted.
    ns['_publish_hil_gps'](bridge, fix)
    assert bridge._mav.mav.hil_gps_send.call_args.args[5] == 37
