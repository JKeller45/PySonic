from attrs import define, validators, field
import numpy.typing as npt
import sys
import threading

class traced_thread(threading.Thread):
  def __init__(self, *args, **keywords):
    threading.Thread.__init__(self, *args, **keywords)
    self.killed = False
 
  def start(self):
    self.__run_backup = self.run
    self.run = self.__run     
    threading.Thread.start(self)
 
  def __run(self):
    sys.settrace(self.globaltrace)
    self.__run_backup()
    self.run = self.__run_backup
 
  def globaltrace(self, frame, event, arg):
    if event == 'call':
      return self.localtrace
    else:
      return None
 
  def localtrace(self, frame, event, arg):
    if self.killed:
      if event == 'line':
        raise SystemExit()
    return self.localtrace
  
  def kill(self):
    self.killed = True

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