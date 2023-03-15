from attrs import define, validators, field
import numpy.typing as npt
import sys
import threading
import ctypes

class thread_with_exception(threading.Thread):
    def __init__(self, *args, **kwargs):
        threading.Thread.__init__(self, *args, **kwargs)
          
    def get_id(self):
        # returns id of the respective thread
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id
  
    def raise_exception(self):
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id,ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print('Exception raise failure')

class Progress_Spoof:
    def step(self, *args):
        pass

class Main_Spoof:
    def update(self):
        pass

@define
class Settings:
    audio_file: str = field(validator=validators.instance_of(str))
    output: str = field(validator=validators.instance_of(str))
    length: float = field(converter=float, validator=validators.instance_of(float))
    size: list[int] = field(validator=validators.instance_of(list))
    color: list[int] = field(validator=validators.instance_of(list))
    background: npt.ArrayLike | str = field()
    frame_rate: int = field(validator=validators.instance_of(int))
    width: int = field(validator=validators.instance_of(int))
    separation: int = field(validator=validators.instance_of(int))
    position: str = field(validator=validators.instance_of(str))
    SSAA: bool = field(validator=validators.instance_of(bool))
    AISS: bool = field(validator=validators.instance_of(bool))
    solar: bool = field(validator=validators.instance_of(bool))
    wave: bool = field(validator=validators.instance_of(bool))
    use_gpu: bool = field(validator=validators.instance_of(bool))
    memory_compression: bool = field(validator=validators.instance_of(bool))
    circular_looped_video: bool = field(validator=validators.instance_of(bool))