import threading
from contextlib import contextmanager
import rmm

# Global state for RMM management
class RMMManager:
    _lock = threading.Lock()
    _instances = {}
    _ref_counts = {}
    _initial_pool_size = 2**30  # 1GB

    @classmethod
    def initialize_pool(cls, device_id: int, pool_size: int = None):
        with cls._lock:
            pool_size = pool_size or cls._initial_pool_size
            if device_id not in cls._instances:
                pool = rmm.mr.PoolMemoryResource(
                    rmm.mr.CudaMemoryResource(), 
                    initial_pool_size=pool_size
                )
                rmm.mr.set_per_device_resource(device_id, pool)
                cls._instances[device_id] = pool
                cls._ref_counts[device_id] = 0
            cls._ref_counts[device_id] += 1

    @classmethod
    def release_pool(cls, device_id: int):
        with cls._lock:
            if device_id in cls._ref_counts:
                cls._ref_counts[device_id] -= 1
                if cls._ref_counts[device_id] <= 0:
                    # Reset to default memory resource
                    rmm.mr.set_per_device_resource(
                        device_id, 
                        rmm.mr.CudaMemoryResource()
                    )
                    del cls._instances[device_id]
                    del cls._ref_counts[device_id]

    @classmethod
    @contextmanager
    def rmm_context(cls, device_id: int, pool_size: int = None):
        try:
            cls.initialize_pool(device_id, pool_size)
            yield
        finally:
            cls.release_pool(device_id)