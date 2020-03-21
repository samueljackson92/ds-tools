import os
import platform
import psutil
import socket
import cpuinfo
import pynvml as nv

from collections import defaultdict
from threading import Timer
from abc import abstractmethod, ABCMeta

class RepeatedTimer:
    __metaclass__ = ABCMeta

    def __init__(self, interval, *args, **kwargs):
        self._timer     = None
        self.interval   = interval
        self.args       = args
        self.kwargs     = kwargs
        self.is_running = False

    @abstractmethod
    def run(self):
        pass

    def _run(self):
        self.is_running = False
        self.start()
        self.run(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False

    def __enter__(self):
        self.start()

    def __exit__(self ,type, value, traceback):
        self.stop()

    def __deepcopy__(self, memo):
        return RepeatedTimer(interval=self.interval, *self.args, **self.kwargs)


def bytes_to(byte_count, to, bsize=1024):
    size = {'k' : 1, 'm': 2, 'g' : 3, 't' : 4, 'p' : 5, 'e' : 6 }
    r = float(byte_count)
    for i in range(size[to]):
        r = r / bsize
    return r

class HostSpec:

    def __init__(self, per_device=False):
        self._per_device = per_device

    @property
    def name(self):
        return os.name

    @property
    def system(self):
        return platform.system()

    @property
    def node_name(self):
        return socket.gethostname()

    @property
    def ip_address(self):
        return socket.gethostbyname(self.node_name)

    @property
    def release(self):
        return platform.release()

    @property
    def num_cores(self):
        return psutil.cpu_count()

    @property
    def total_memory(self):
        mem = psutil.virtual_memory()
        return mem.total

    @property
    def cpu_percent(self):
        info = psutil.cpu_percent(percpu=self._per_device)
        if not self._per_device:
            info = [info]

        metrics = {}
        for i, percent in enumerate(info):
            metrics['cpu_{}_utilization'.format(i)] = percent
        return metrics

    @property
    def cpu_info(self):
        info = cpuinfo.get_cpu_info()
        keys = ['brand', 'arch', 'vendor_id', 'hz_advertised', 'hz_actual', 'model', 'family']
        return {'cpu_' + key: value for key, value in info.items() if key in keys}

    @property
    def disk_io(self):
        info = psutil.disk_io_counters(perdisk=self._per_device)
        if self._per_device:
            return {'disk_' + key + '_' + k: v for key, value in info.items() for k, v in value._asdict().items()}
        else:
            return {'disk_' + k: v for k, v in info._asdict().items()}

    @property
    def net_io(self):
        info = psutil.net_io_counters(pernic=self._per_device)
        if self._per_device:
            return {'net_' + key + '_' + k: v for key, value in info.items() for k, v in value._asdict().items()}
        else:
            return {'net_' + k: v for k, v in info._asdict().items()}

    @property
    def memory(self):
        memory_props = dict(psutil.virtual_memory()._asdict())
        host_memory = dict({"memory_" + key: value for key, value in memory_props.items()})

        metrics = {}
        metrics['host_memory_free'] = bytes_to(host_memory['memory_free'], 'm')
        metrics['host_memory_used'] = bytes_to(host_memory['memory_used'], 'm')
        metrics['host_memory_available'] = bytes_to(host_memory['memory_available'], 'm')
        metrics['host_memory_utilization'] = host_memory['memory_percent']
        return metrics

class DeviceSpecs:

    def __init__(self):
        nv.nvmlInit()
        self._device_count = nv.nvmlDeviceGetCount()
        self._specs = [DeviceSpec(i) for i in range(self.device_count)]

    @property
    def device_count(self):
        return self._device_count

    @property
    def uuids(self):
        return {'gpu_{}_uuid'.format(i): spec.uuid for i, spec in enumerate(self._specs)}

    @property
    def names(self):
        return {'gpu_{}_name'.format(i): spec.name for i, spec in enumerate(self._specs)}

    @property
    def brands(self):
        return {'gpu_{}_brand'.format(i): spec.brand for i, spec in enumerate(self._specs)}

    @property
    def minor_numbers(self):
        return {'gpu_{}_minor_number'.format(i): spec.minor_number for i, spec in enumerate(self._specs)}

    @property
    def is_multi_gpu_board(self):
        return {'gpu_{}_is_mulitgpu_board'.format(i): spec.is_multigpu_board for i, spec in enumerate(self._specs)}

    @property
    def power_usage(self):
        return {'gpu_{}_power'.format(i): spec.power_usage for i, spec in enumerate(self._specs)}

    @property
    def memory(self):
        memory_info = {}
        for i, spec in enumerate(self._specs):
            for key, value in spec.memory.items():
                memory_info['gpu_{}_memory_{}'.format(i, key)] = value
        return memory_info

    @property
    def utilization_rates(self):
        memory_info = {}
        for i, spec in enumerate(self._specs):
            for key, value in spec.utilization_rates.items():
                memory_info['gpu_{}_{}'.format(i, key)] = value
        return memory_info

class DeviceSpec:

    def __init__(self, index):
        nv.nvmlInit()
        self._handle = nv.nvmlDeviceGetHandleByIndex(index)

    @property
    def uuid(self):
        """ NVIDIA device UUID """
        return nv.nvmlDeviceGetUUID(self._handle).decode()

    @property
    def name(self):
        """ NVIDIA device name """
        return nv.nvmlDeviceGetName(self._handle).decode()

    @property
    def brand(self):
        """ Device brand name as a string

        This function maps the device code to a string representation using the
        following enum:

            NVML_BRAND_UNKNOWN = 0
            NVML_BRAND_QUADRO = 1
            NVML_BRAND_TESLA = 2
            NVML_BRAND_NVS = 3
            NVML_BRAND_GRID = 4
            NVML_BRAND_GEFORCE = 5
            NVML_BRAND_TITAN = 6
        """
        brand_enum = nv.nvmlDeviceGetBrand(self._handle)

        if brand_enum == 1:
            return 'Quadro'
        elif brand_enum == 2:
            return 'Tesla'
        elif brand_enum == 3:
            return 'NVS'
        elif brand_enum == 4:
            return 'Grid'
        elif brand_enum == 5:
            return 'GeForce'
        elif brand_enum == 6:
            return 'Titan'
        else:
            return 'Unknown'

    @property
    def minor_number(self):
        return nv.nvmlDeviceGetMinorNumber(self._handle)

    @property
    def is_multi_gpu_board(self):
        return nv.nvmlDeviceGetMultiGpuBoard(self._handle)

    @property
    def utilization_rates(self):
        rates = nv.nvmlDeviceGetUtilizationRates(self._handle)
        return dict(gpu=rates.gpu, memory=rates.memory)

    @property
    def memory(self):
        """ Total, free, and used memory in bytes"""
        info = nv.nvmlDeviceGetMemoryInfo(self._handle)
        return dict(free=info.free, total=info.total, used=info.used)

    @property
    def power_usage(self):
        """ Power usage for the device in milliwatts

        From the NVIDIA documentation:
         - On Fermi and Kepler GPUs the reading is accurate to within +/- 5% of current power draw.
        """
        return nv.nvmlDeviceGetPowerUsage(self._handle)

class DeviceLogger(RepeatedTimer):

    def __init__(self, *args, **kwargs):
        super(DeviceLogger, self).__init__(*args, **kwargs)
        self._spec = DeviceSpecs()
        self._metrics = defaultdict(list)

    def run(self):
        for key, value in self._spec.power_usage.items():
            self._metrics[key].append(value)

    @property
    def metrics(self):
        return self._metrics
