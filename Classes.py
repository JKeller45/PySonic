from attrs import define, validators, field
import numpy.typing as npt

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