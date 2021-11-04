import numpy as np


class SignalGeneratorRT:
    """Generates signals based on amplitude, frequency, and offset.
    
    Class is instantiated by passing in amplitude, frequency, and offset.
    For random signals, amplitude is interpreted as the mean and offset 
    as the standard deviation.
    For step signals, frequency is used as the step location.
    For the ramp signal, amplitude is interpreted as the final value.
    """

    def __init__(self, amplitude=1.0, frequency=0.001, y_offset=0):
        self.amplitude = float(amplitude)
        self.frequency = float(frequency)
        self.y_offset = float(y_offset)

    def square(self, t):
        """Generates a square wave signal."""
        if t % (1.0 / self.frequency) <= 0.5 / self.frequency:
            out = self.amplitude + self.y_offset
        else:
            out = -self.amplitude + self.y_offset
        return out

    def sawtooth(self, t):
        """Generates a sawtooth signal."""
        tmp = t % (0.5 / self.frequency)
        out = 4 * self.amplitude * self.frequency * tmp - self.amplitude + self.y_offset
        return out

    def step(self, t):
        """Generates a step signal."""
        if t >= self.frequency:
            out = self.amplitude + self.y_offset
        else:
            out = self.y_offset
        return out

    def random(self, t):
        """Generates a random value from a Gaussian distribution."""
        out = np.random.normal(self.amplitude, self.y_offset)
        return out

    def sin(self, t, k):
        """Generates a sine wave signal."""
        return (
            self.amplitude * np.sin(2 * np.pi * self.frequency * t[k]) + self.y_offset
        )

    def ramp(self, t, k):
        """Generates a ramp signal."""
        return self.amplitude / t[-1] * t[k]


class SignalGeneratorFunction:
    """Generates signals based on amplitude, frequency, and offset.
    
    Class is instantiated by passing in amplitude, frequency, and offset.
    For random signals, amplitude is interpreted as the mean and offset 
    as the standard deviation.
    For step signals, frequency is used as the step location.
    For the ramp signal, amplitude is interpreted as the final value.
    """

    def __init__(self, amplitude=1.0, frequency=0.001, y_offset=0):
        self.amplitude = float(amplitude)
        self.frequency = float(frequency)
        self.y_offset = float(y_offset)

    # def square(self, t):
    #     """Generates a square wave signal."""
    #     if t % (1.0 / self.frequency) <= 0.5 / self.frequency:
    #         out = self.amplitude + self.y_offset
    #     else:
    #         out = -self.amplitude + self.y_offset
    #     return out

    # def sawtooth(self, t):
    #     """Generates a sawtooth signal."""
    #     tmp = t % (0.5 / self.frequency)
    #     out = 4 * self.amplitude * self.frequency * tmp - self.amplitude + self.y_offset
    #     return out

    def step(self, t):
        """Generates a step signal."""
        x = np.zeros(t.shape)
        for k, l in enumerate(t):
            if l >= self.frequency:
                x[k] = self.amplitude + self.y_offset
            else:
                x[k] = self.y_offset
        return x

    def random(self, t):
        """Generates a random signal from a Gaussian distribution."""
        out = np.random.normal(self.amplitude, self.y_offset, size=t.size)
        return out

    def sin(self, t):
        """Generates a sine wave signal."""
        x = self.amplitude * np.sin(2 * np.pi * self.frequency * t) + self.y_offset
        return x

    def ramp(self, t):
        """Generates a ramp signal."""
        x = t.copy()
        for k in range(len(x)):
            x[k] = self.amplitude / t[-1] * t[k] + self.y_offset
        return x
