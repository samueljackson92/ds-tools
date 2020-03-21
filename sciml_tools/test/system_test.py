import pytest
import os
import platform
import psutil
import pynvml as nv
import time
from copy import deepcopy
from sciml_tools.system import HostSpec, DeviceSpec, DeviceSpecs, DeviceLogger

@pytest.fixture(scope="module")
def device_count():
    nv.nvmlInit()
    return nv.nvmlDeviceGetCount()

def test_DeviceLogger(mocker, device_count):

    logger = DeviceLogger(interval=0.1)

    run_stub = mocker.spy(logger, 'run')

    logger.start()
    time.sleep(1)
    logger.stop()

    run_stub.assert_called()

    assert logger.metrics is not None
    assert isinstance(logger.metrics, dict)
    assert len(logger.metrics) == device_count

    if device_count > 0:
        assert 'gpu_0_power' in logger.metrics
        assert isinstance(logger.metrics['gpu_0_power'], list)
        assert len(logger.metrics['gpu_0_power']) > 0

    logger2 = deepcopy(logger)
    assert logger.interval == logger2.interval

def test_host_spec():
    spec = HostSpec()

    assert isinstance(spec.disk_io, dict)
    assert isinstance(spec.net_io, dict)
    assert spec.system == platform.system()
    assert spec.name == os.name
    assert spec.release == platform.release()
    assert spec.num_cores == psutil.cpu_count()
    assert spec.total_memory == psutil.virtual_memory().total


@pytest.mark.skipif(not device_count,
                    reason="requires GPU")
def test_device_spec():
    spec = DeviceSpec(0)

    assert spec.is_multi_gpu_board == False

def test_device_specs(device_count):
    spec = DeviceSpecs()
    assert spec.device_count == device_count
